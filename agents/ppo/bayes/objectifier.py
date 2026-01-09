from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy import ndimage


@dataclass(frozen=True, slots=True)
class GridObject:
    """A single connected component in the discrete grid."""

    obj_id: int
    color: int
    pixels_yx: np.ndarray  # (N,2) int32: [y,x]
    y0: int
    x0: int
    y1: int
    x1: int
    area: int
    cy: float
    cx: float


@dataclass(frozen=True, slots=True)
class ObjectState:
    """Object-centric view of an observation."""

    grid: np.ndarray  # (H,W) uint8
    cursor_y: int
    cursor_x: int
    objects: tuple[GridObject, ...]


def _as_hwc(obs: np.ndarray) -> np.ndarray:
    """Return obs as (H,W,C). Supports (H,W,C) or (C,H,W)."""
    if obs.ndim != 3:
        raise ValueError(f"Expected obs with 3 dims, got shape={obs.shape}")
    # HWC
    if obs.shape[-1] == 10:
        return obs
    # CHW
    if obs.shape[0] == 10:
        return np.transpose(obs, (1, 2, 0))
    raise ValueError(f"Unrecognized obs shape={obs.shape}; expected HWC or CHW with 10 channels")


def _to_uint8_like(x: np.ndarray) -> np.ndarray:
    """Best-effort conversion to uint8 in [0..255]."""
    if x.dtype == np.uint8:
        return x
    # Common case: float32/float64 normalized to [0,1]
    if np.issubdtype(x.dtype, np.floating):
        y = np.clip(x, 0.0, 1.0) * 255.0
        return (y + 0.5).astype(np.uint8)
    # Integer but not uint8
    y = np.clip(x, 0, 255)
    return y.astype(np.uint8)


def extract_cursor_yx(obs: np.ndarray, cursor_channel: int = 2) -> tuple[int, int]:
    """Extract cursor (y,x) from the obs.

    Preferred: cursor marker channel (binary map).
    Fallback: broadcast cursor_x/cursor_y scalar channels (4,5) if marker absent.
    """
    obs_hwc = _as_hwc(obs)
    h, w, _ = obs_hwc.shape

    cursor_map = obs_hwc[:, :, cursor_channel]
    if cursor_map.size > 0 and float(cursor_map.max()) > 0.0:
        flat_idx = int(np.argmax(cursor_map))
        cy, cx = np.unravel_index(flat_idx, (h, w))
        return int(cy), int(cx)

    # Fallback: channels 4 and 5 are broadcast 0..255 scalars.
    cx_val = float(np.mean(obs_hwc[:, :, 4]))
    cy_val = float(np.mean(obs_hwc[:, :, 5]))
    cx = int(round((np.clip(cx_val, 0.0, 255.0) / 255.0) * float(max(1, w - 1))))
    cy = int(round((np.clip(cy_val, 0.0, 255.0) / 255.0) * float(max(1, h - 1))))
    return int(np.clip(cy, 0, h - 1)), int(np.clip(cx, 0, w - 1))


def extract_grid(obs: np.ndarray, grid_channel: int = 0) -> np.ndarray:
    """Extract the discrete grid channel as uint8 (H,W)."""
    obs_hwc = _as_hwc(obs)
    grid = obs_hwc[:, :, grid_channel]
    return _to_uint8_like(grid)


def connected_components(
    grid: np.ndarray,
    *,
    background_colors: Optional[set[int]] = None,
    min_area: int = 1,
    connectivity: int = 4,
) -> tuple[GridObject, ...]:
    """Extract connected components per color as `GridObject`s.

    - **background_colors**: colors to ignore as 'background' components (default: {0})
    - **connectivity**: 4 or 8
    """
    if grid.ndim != 2:
        raise ValueError(f"Expected grid (H,W), got shape={grid.shape}")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if min_area < 1:
        raise ValueError("min_area must be >= 1")

    bg = background_colors if background_colors is not None else {0}
    grid_u8 = _to_uint8_like(grid)

    if connectivity == 4:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    else:
        structure = np.ones((3, 3), dtype=bool)

    obj_id = 0
    out: list[GridObject] = []

    for color in np.unique(grid_u8):
        c = int(color)
        if c in bg:
            continue
        mask = grid_u8 == c
        if not bool(mask.any()):
            continue
        labeled, n = ndimage.label(mask, structure=structure)
        for k in range(1, int(n) + 1):
            ys_xs = np.argwhere(labeled == k).astype(np.int32)
            area = int(ys_xs.shape[0])
            if area < min_area:
                continue
            y0 = int(ys_xs[:, 0].min())
            x0 = int(ys_xs[:, 1].min())
            y1 = int(ys_xs[:, 0].max())
            x1 = int(ys_xs[:, 1].max())
            cy = float(ys_xs[:, 0].mean())
            cx = float(ys_xs[:, 1].mean())
            out.append(
                GridObject(
                    obj_id=obj_id,
                    color=c,
                    pixels_yx=ys_xs,
                    y0=y0,
                    x0=x0,
                    y1=y1,
                    x1=x1,
                    area=area,
                    cy=cy,
                    cx=cx,
                )
            )
            obj_id += 1

    return tuple(out)


def objectify_obs(
    obs: np.ndarray,
    *,
    grid_channel: int = 0,
    cursor_channel: int = 2,
    background_colors: Optional[set[int]] = None,
    min_area: int = 1,
    connectivity: int = 4,
) -> ObjectState:
    """Convert a 10-channel observation tensor into an object-centric state."""
    grid = extract_grid(obs, grid_channel=grid_channel)
    cy, cx = extract_cursor_yx(obs, cursor_channel=cursor_channel)
    objs = connected_components(
        grid,
        background_colors=background_colors,
        min_area=min_area,
        connectivity=connectivity,
    )
    return ObjectState(grid=grid, cursor_y=cy, cursor_x=cx, objects=objs)


def objects_to_mask(objects: Sequence[GridObject], shape_hw: tuple[int, int]) -> np.ndarray:
    """Render objects into a labeled mask (H,W) where each object id+1 marks its pixels."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.int32)
    for obj in objects:
        pts = obj.pixels_yx
        if pts.size == 0:
            continue
        mask[pts[:, 0], pts[:, 1]] = int(obj.obj_id) + 1
    return mask


def compute_object_adjacency(
    objects: Sequence[GridObject],
    *,
    max_manhattan_dist: int = 1,
) -> set[tuple[int, int]]:
    """Compute undirected adjacency edges between objects based on pixel proximity."""
    if max_manhattan_dist < 1:
        raise ValueError("max_manhattan_dist must be >= 1")
    if not objects:
        return set()

    # Build a dense owner map: each pixel -> object id+1 (0 for none). Then dilate each object and test overlap.
    h = max((o.y1 for o in objects), default=0) + 1
    w = max((o.x1 for o in objects), default=0) + 1
    owner = objects_to_mask(objects, (h, w))

    edges: set[tuple[int, int]] = set()
    # Use a small manhattan ball for dilation.
    r = int(max_manhattan_dist)
    struct = np.zeros((2 * r + 1, 2 * r + 1), dtype=bool)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if abs(dy) + abs(dx) <= r:
                struct[dy + r, dx + r] = True

    for obj in objects:
        pts = obj.pixels_yx
        if pts.size == 0:
            continue
        local = np.zeros_like(owner, dtype=bool)
        local[pts[:, 0], pts[:, 1]] = True
        dil = ndimage.binary_dilation(local, structure=struct)
        neigh_ids = set(int(x) for x in np.unique(owner[dil]) if int(x) != 0)
        for nid in neigh_ids:
            other_id = nid - 1
            if other_id == obj.obj_id:
                continue
            a, b = sorted((int(obj.obj_id), int(other_id)))
            edges.add((a, b))
    return edges



