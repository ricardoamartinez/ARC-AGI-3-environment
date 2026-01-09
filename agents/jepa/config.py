"""
V-JEPA 2 RL Configuration.

Loads TOML config and maps to environment variables for the JEPA agent.
Note: Some env vars retain legacy JEPA_ prefix for backward compatibility.
These will be gradually migrated to JEPA_ prefix.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _read_toml(path: Path) -> dict[str, Any]:
    # Python 3.11+ built-in TOML parser: tomllib.
    # For older Python versions, we *optionally* fall back to tomli if installed.
    # If neither is available, we skip config loading (and rely on env vars).
    try:
        import tomllib  # type: ignore
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError:
            print(
                "[JEPAConfig] WARNING: TOML parser not available (need Python 3.11+ 'tomllib' or 'tomli'). "
                f"Skipping config file: {path}",
                flush=True,
            )
            return {}

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("TOML root must be a table")
    return data  # type: ignore[return-value]


def _flatten(prefix: str, d: dict[str, Any], out: dict[str, Any]) -> None:
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            _flatten(key, v, out)
        else:
            out[key] = v


def _as_env_value(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)


@dataclass(frozen=True)
class JEPAConfig:
    """V-JEPA 2 RL configuration loaded from TOML."""
    raw: dict[str, Any]

    @staticmethod
    def load(path: Path) -> "JEPAConfig":
        return JEPAConfig(raw=_read_toml(path))


def apply_jepa_config(
    config: JEPAConfig,
    *,
    override_env: bool = False,
    env: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """
    Apply TOML config values by writing to environment variables.

    - If override_env=False (default), existing env vars win.
    - Returns the env var updates applied (for logging/debug).
    
    Note: Many env vars retain legacy JEPA_ prefix for backward compatibility.
    """
    if env is None:
        env = os.environ  # type: ignore[assignment]

    flat: dict[str, Any] = {}
    _flatten("", config.raw, flat)

    # Map TOML keys -> env vars (some retain legacy JEPA_ prefix for compatibility)
    key_map: dict[str, str] = {
        # Trainer selection
        "trainer.name": "JEPA_TRAINER",
        "trainer.action_mode": "JEPA_ACTION_MODE",
        "trainer.require_goal": "JEPA_REQUIRE_GOAL",
        "trainer.allow_game_actions": "JEPA_ALLOW_GAME_ACTIONS",

        # Feature mode
        "features.mode": "JEPA_FEATURES",
        "features.dim": "JEPA_FEATURES_DIM",

        # Reward / shaping
        "reward.mode": "JEPA_REWARD_MODE",
        "reward.scale": "JEPA_REWARD_SCALE",
        "reward.goal_radius": "JEPA_GOAL_RADIUS",
        "reward.arrival_bonus": "JEPA_ARRIVAL_BONUS",
        "reward.action_l2_penalty": "JEPA_ACTION_L2_PENALTY",
        "reward.vel_l2_penalty": "JEPA_VEL_L2_PENALTY",
        "reward.jerk_l2_penalty": "JEPA_JERK_L2_PENALTY",
        "reward.wall_penalty": "JEPA_WALL_PENALTY",

        # Physics / control
        "physics.bounds": "JEPA_BOUNDS",
        "physics.max_speed": "JEPA_MAX_SPEED",
        "physics.inertia_alpha": "JEPA_INERTIA_ALPHA",
        "physics.vel_response": "JEPA_VEL_RESPONSE",
        "physics.damping": "JEPA_DAMPING",
        "physics.pd_kp": "JEPA_PD_KP",
        "physics.pd_kd": "JEPA_PD_KD",
        "physics.stop_radius": "JEPA_STOP_RADIUS",
        "physics.stop_speed": "JEPA_STOP_SPEED",
        "physics.speed_scale_max": "JEPA_SPEED_SCALE_MAX",
        "physics.target_ema_alpha": "JEPA_TARGET_EMA_ALPHA",
        "physics.target_delta_max": "JEPA_TARGET_DELTA_MAX",

        # SAC hyperparams
        "sac.lr": "JEPA_SAC_LR",
        "sac.gamma": "JEPA_SAC_GAMMA",
        "sac.tau": "JEPA_SAC_TAU",
        "sac.batch_size": "JEPA_SAC_BATCH_SIZE",
        "sac.buffer_size": "JEPA_SAC_BUFFER_SIZE",
        "sac.warmup": "JEPA_SAC_WARMUP",
        "sac.updates_per_step": "JEPA_SAC_UPDATES_PER_STEP",
        "sac.auto_alpha": "JEPA_SAC_AUTO_ALPHA",
        "sac.target_entropy": "JEPA_SAC_TARGET_ENTROPY",
        "sac.alpha": "JEPA_SAC_ALPHA",
        "sac.features": "JEPA_SAC_FEATURES",

        # RTAC hyperparams (online TD actor-critic)
        "rtac.lr": "JEPA_LR",
        "rtac.gamma": "JEPA_GAMMA",
        "rtac.td_lambda": "JEPA_TD_LAMBDA",
        "rtac.value_coef": "JEPA_VALUE_COEF",
        "rtac.entropy_coef": "JEPA_ENTROPY_COEF",
        "rtac.adv_clip": "JEPA_ADV_CLIP",
        "rtac.max_grad_norm": "JEPA_MAX_GRAD_NORM",
        "rtac.device": "JEPA_DEVICE",

        # Logging
        "logging.log_every": "JEPA_LOG_EVERY",
        "logging.ui_fps": "JEPA_UI_FPS",
        "logging.ui_grid_fps": "JEPA_UI_GRID_FPS",

        # Intrinsic motivation (exploration in sparse reward settings)
        "intrinsic.enabled": "JEPA_INTRINSIC_ENABLED",
        "intrinsic.use_rnd": "JEPA_USE_RND",
        "intrinsic.use_counts": "JEPA_USE_COUNTS",
        "intrinsic.rnd_scale": "JEPA_RND_SCALE",
        "intrinsic.count_scale": "JEPA_COUNT_SCALE",
        "intrinsic.rnd_lr": "JEPA_RND_LR",

        # Sparse reward mode
        "reward.sparse_mode": "JEPA_SPARSE_REWARD",
        "reward.success_bonus": "JEPA_SUCCESS_BONUS",

        # V-JEPA World Model
        "jepa.enabled": "JEPA_USE_ACTION_JEPA",
        "jepa.bias": "JEPA_ACTION_BIAS",
        "jepa.train_every": "JEPA_TRAIN_EVERY",
        "jepa.warmup": "JEPA_WARMUP",
        "jepa.show_viz": "JEPA_SHOW_VIZ",
        
        # World Model (legacy)
        "world_model.latent_dim": "WM_LATENT_DIM",
        "world_model.planning_horizon": "WM_PLANNING_HORIZON",
        "world_model.use_planner": "WM_USE_PLANNER",
        "world_model.use_curiosity": "WM_USE_CURIOSITY",
        "world_model.curiosity_scale": "WM_CURIOSITY_SCALE",
        "world_model.train_every": "WM_TRAIN_EVERY",
        "world_model.exploration_steps": "WM_EXPLORATION_STEPS",
        "world_model.buffer_size": "WM_BUFFER_SIZE",
        "world_model.batch_size": "WM_BATCH_SIZE",
        "world_model.lr": "WM_LR",
        "world_model.mask_ratio": "WM_MASK_RATIO",
        "world_model.mask_coef": "WM_MASK_COEF",
        "world_model.variance_coef": "WM_VAR_COEF",
        "world_model.variance_target": "WM_VAR_TARGET",
        "world_model.mark_cursor": "WM_MARK_CURSOR",
        "world_model.cursor_token": "WM_CURSOR_TOKEN",
        
        # V-JEPA 2 style settings
        "world_model.rollout_steps": "WM_ROLLOUT_STEPS",
        "world_model.rollout_coef": "WM_ROLLOUT_COEF",
        "world_model.use_patch_tokens": "WM_USE_PATCH_TOKENS",
        "world_model.predictor_layers": "WM_PREDICTOR_LAYERS",
        "world_model.predictor_heads": "WM_PREDICTOR_HEADS",
    }

    applied: dict[str, str] = {}
    for toml_key, env_key in key_map.items():
        if toml_key not in flat:
            continue
        if (not override_env) and env_key in env:
            continue
        env_val = _as_env_value(flat[toml_key])
        env[env_key] = env_val
        applied[env_key] = env_val

    return applied


def apply_jepa_config_if_present(
    *,
    override_env: bool = False,
    config_path: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """
    Convenience:
    - Reads TOML from JEPA_CONFIG (or `jepa_config.toml` at repo root)
    - Applies it if the file exists
    """
    if config_path is None:
        config_path = os.environ.get("JEPA_CONFIG", "jepa_config.toml")
    path = Path(config_path)
    if not path.is_file():
        return {}
    cfg = JEPAConfig.load(path)
    return apply_jepa_config(cfg, override_env=override_env, env=env)
