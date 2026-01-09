import numpy as np
import pytest

from agents.ppo.bayes.objectifier import connected_components, extract_cursor_yx, objectify_obs


@pytest.mark.unit
class TestObjectifier:
    def test_connected_components_4conn_splits_diagonal(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[1, 1] = 3
        grid[2, 2] = 3  # diagonal to (1,1)
        objs = connected_components(grid, background_colors={0}, connectivity=4)
        assert len(objs) == 2
        assert {o.color for o in objs} == {3}

    def test_connected_components_8conn_merges_diagonal(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[1, 1] = 3
        grid[2, 2] = 3  # diagonal to (1,1)
        objs = connected_components(grid, background_colors={0}, connectivity=8)
        assert len(objs) == 1
        assert objs[0].area == 2

    def test_extract_cursor_marker(self):
        obs = np.zeros((5, 5, 10), dtype=np.uint8)
        obs[4, 2, 2] = 255  # cursor channel
        cy, cx = extract_cursor_yx(obs)
        assert (cy, cx) == (4, 2)

    def test_extract_cursor_fallback_scalar(self):
        obs = np.zeros((5, 5, 10), dtype=np.uint8)
        # broadcast cursor_x=255 => x=4, cursor_y=0 => y=0
        obs[:, :, 4] = 255
        obs[:, :, 5] = 0
        cy, cx = extract_cursor_yx(obs)
        assert (cy, cx) == (0, 4)

    def test_objectify_handles_chw_float(self):
        # Build an HWC uint8 obs then convert to CHW float [0,1]
        obs = np.zeros((6, 6, 10), dtype=np.uint8)
        obs[2:4, 1:3, 0] = 7  # a 2x2 object
        obs[3, 5, 2] = 255  # cursor marker
        chw = np.transpose(obs, (2, 0, 1)).astype(np.float32) / 255.0
        st = objectify_obs(chw, background_colors={0})
        assert (st.cursor_y, st.cursor_x) == (3, 5)
        assert len(st.objects) == 1
        assert st.objects[0].color == 7
        assert st.objects[0].area == 4



