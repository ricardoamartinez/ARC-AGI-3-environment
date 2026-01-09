import numpy as np
import pytest
from types import SimpleNamespace

from agents.ppo.envs.arc_env import ARCGymEnv
from agents.ppo.world_model import WorldModel
from agents.structs import FrameData, GameState


@pytest.mark.unit
def test_world_model_train_step_mask_and_variance() -> None:
    wm = WorldModel(
        device="cpu",
        continuous_action_dim=2,
        num_discrete_actions=11,
        use_planner=False,
        batch_size=2,
        mask_ratio=0.5,
        mask_loss_coef=1.0,
        variance_coef=1.0,
        variance_target=0.1,
    )

    s = np.random.randint(0, 10, (64, 64), dtype=np.uint8)
    ns = np.random.randint(0, 10, (64, 64), dtype=np.uint8)

    wm.add_transition(
        state=s,
        cont_action=np.array([0.1, -0.2], dtype=np.float32),
        disc_action=0,
        next_state=ns,
        reward=0.0,
        done=False,
        win=False,
    )
    wm.add_transition(
        state=ns,
        cont_action=np.array([0.0, 0.0], dtype=np.float32),
        disc_action=1,
        next_state=s,
        reward=1.0,
        done=True,
        win=True,
    )

    stats = wm.train_step()
    assert "predictor_loss" in stats
    assert "win_loss" in stats
    assert "mask_loss" in stats
    assert "latent_var_mean" in stats


@pytest.mark.unit
def test_arc_env_sparse_reward_override_and_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid any API calls: reuse a dummy FrameData as the "latest frame".
    monkeypatch.setenv("PPO_DISABLE_RESET_BOOTSTRAP", "1")
    monkeypatch.setenv("PPO_ALLOW_GAME_ACTIONS", "0")
    monkeypatch.setenv("PPO_ACTION_MODE", "delta")

    monkeypatch.setenv("PPO_SPARSE_REWARD", "1")
    monkeypatch.setenv("PPO_SUCCESS_BONUS", "7.0")

    grid = np.zeros((64, 64), dtype=np.uint8)
    frame = FrameData(
        game_id="dummy",
        frame=[grid.tolist()],
        state=GameState.NOT_FINISHED,
        score=0,
    )
    agent = SimpleNamespace(
        game_id="dummy",
        cursor_x=32.0,
        cursor_y=32.0,
        manual_dopamine=0.0,
        manual_pain=0.0,
        spatial_goal=None,
        goal_version=0,
        goal_shaping_enabled=False,
        frames=[frame],
    )

    env = ARCGymEnv(agent, max_steps=10)
    env.reset()

    # Execute a discrete action by setting trigger high.
    act = (np.array([0.0, 0.0, 1.0], dtype=np.float32), 4)
    _obs1, reward, _terminated, _truncated, info = env.step(act)

    assert reward == 0.0  # sparse reward override: not a win
    assert info.get("final_action_idx") == 4
    assert info.get("game_success") is False
    assert "NOT_FINISHED" in str(info.get("game_state"))

