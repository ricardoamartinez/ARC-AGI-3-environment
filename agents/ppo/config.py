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
            # Keep startup robust even if user runs with older Python.
            # main.py calls this before logging is configured, so use print.
            print(
                "[PPOConfig] WARNING: TOML parser not available (need Python 3.11+ 'tomllib' or 'tomli'). "
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
class PPOConfig:
    raw: dict[str, Any]

    @staticmethod
    def load(path: Path) -> "PPOConfig":
        return PPOConfig(raw=_read_toml(path))


def apply_ppo_config(
    config: PPOConfig,
    *,
    override_env: bool = False,
    env: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """
    Apply TOML config values by writing to environment variables.

    - If override_env=False (default), existing env vars win.
    - Returns the env var updates applied (for logging/debug).
    """
    if env is None:
        env = os.environ  # type: ignore[assignment]

    flat: dict[str, Any] = {}
    _flatten("", config.raw, flat)

    # Map TOML keys -> existing PPO_* env vars already used throughout the codebase.
    key_map: dict[str, str] = {
        # Trainer selection
        "trainer.name": "PPO_TRAINER",
        "trainer.action_mode": "PPO_ACTION_MODE",
        "trainer.require_goal": "PPO_REQUIRE_GOAL",
        "trainer.allow_game_actions": "PPO_ALLOW_GAME_ACTIONS",

        # Feature mode (shared by RTAC/SAC)
        "features.mode": "PPO_FEATURES",
        "features.dim": "PPO_FEATURES_DIM",

        # Reward / shaping
        "reward.mode": "PPO_REWARD_MODE",
        "reward.scale": "PPO_REWARD_SCALE",
        "reward.goal_radius": "PPO_GOAL_RADIUS",
        "reward.arrival_bonus": "PPO_ARRIVAL_BONUS",
        "reward.action_l2_penalty": "PPO_ACTION_L2_PENALTY",
        "reward.vel_l2_penalty": "PPO_VEL_L2_PENALTY",
        "reward.jerk_l2_penalty": "PPO_JERK_L2_PENALTY",
        "reward.wall_penalty": "PPO_WALL_PENALTY",

        # Physics / control
        "physics.bounds": "PPO_BOUNDS",
        "physics.max_speed": "PPO_MAX_SPEED",
        "physics.inertia_alpha": "PPO_INERTIA_ALPHA",
        "physics.bounds": "PPO_BOUNDS",
        "physics.vel_response": "PPO_VEL_RESPONSE",
        "physics.damping": "PPO_DAMPING",
        "physics.pd_kp": "PPO_PD_KP",
        "physics.pd_kd": "PPO_PD_KD",
        "physics.stop_radius": "PPO_STOP_RADIUS",
        "physics.stop_speed": "PPO_STOP_SPEED",
        "physics.speed_scale_max": "PPO_SPEED_SCALE_MAX",
        "physics.target_ema_alpha": "PPO_TARGET_EMA_ALPHA",
        "physics.target_delta_max": "PPO_TARGET_DELTA_MAX",

        # SAC hyperparams
        "sac.lr": "PPO_SAC_LR",
        "sac.gamma": "PPO_SAC_GAMMA",
        "sac.tau": "PPO_SAC_TAU",
        "sac.batch_size": "PPO_SAC_BATCH_SIZE",
        "sac.buffer_size": "PPO_SAC_BUFFER_SIZE",
        "sac.warmup": "PPO_SAC_WARMUP",
        "sac.updates_per_step": "PPO_SAC_UPDATES_PER_STEP",
        "sac.auto_alpha": "PPO_SAC_AUTO_ALPHA",
        "sac.target_entropy": "PPO_SAC_TARGET_ENTROPY",
        "sac.alpha": "PPO_SAC_ALPHA",
        "sac.features": "PPO_SAC_FEATURES",
        "sac.clear_buffer_on_goal_change": "PPO_SAC_CLEAR_BUFFER_ON_GOAL_CHANGE",
        "sac.goal_change_burst_updates": "PPO_SAC_GOAL_CHANGE_BURST_UPDATES",
        "sac.deterministic_after_steps": "PPO_SAC_DETERMINISTIC_AFTER_STEPS",
        "sac.async_updates": "PPO_SAC_ASYNC_UPDATES",
        "sac.infer_device": "PPO_SAC_INFER_DEVICE",
        "sac.sync_every_updates": "PPO_SAC_SYNC_EVERY_UPDATES",

        # RTAC hyperparams (online TD actor-critic)
        "rtac.lr": "PPO_LR",
        "rtac.gamma": "PPO_GAMMA",
        "rtac.td_lambda": "PPO_TD_LAMBDA",
        "rtac.value_coef": "PPO_VALUE_COEF",
        "rtac.entropy_coef": "PPO_ENTROPY_COEF",
        "rtac.adv_clip": "PPO_ADV_CLIP",
        "rtac.max_grad_norm": "PPO_MAX_GRAD_NORM",
        "rtac.device": "PPO_RTAC_DEVICE",
        "rtac.async_updates": "PPO_RTAC_ASYNC_UPDATES",
        "rtac.infer_device": "PPO_RTAC_INFER_DEVICE",
        "rtac.sync_every_updates": "PPO_RTAC_SYNC_EVERY_UPDATES",
        "rtac.queue_size": "PPO_RTAC_QUEUE_SIZE",
        "rtac.torch_num_threads": "PPO_TORCH_NUM_THREADS",
        "rtac.log_std_min": "PPO_LOG_STD_MIN",
        "rtac.log_std_max": "PPO_LOG_STD_MAX",
        "features.disable_lstm": "PPO_DISABLE_LSTM",

        # Logging
        "logging.log_every": "PPO_LOG_EVERY",
        "logging.ui_fps": "PPO_UI_FPS",
        "logging.ui_grid_fps": "PPO_UI_GRID_FPS",
        "logging.ui_bayes_maps": "PPO_UI_BAYES_MAPS",

        # Intrinsic motivation (exploration in sparse reward settings)
        "intrinsic.enabled": "PPO_INTRINSIC_ENABLED",
        "intrinsic.use_rnd": "PPO_USE_RND",
        "intrinsic.use_counts": "PPO_USE_COUNTS",
        "intrinsic.rnd_scale": "PPO_RND_SCALE",
        "intrinsic.count_scale": "PPO_COUNT_SCALE",
        "intrinsic.rnd_lr": "PPO_RND_LR",

        # Sparse reward mode (only extrinsic reward on game success)
        "reward.sparse_mode": "PPO_SPARSE_REWARD",
        "reward.success_bonus": "PPO_SUCCESS_BONUS",

        # World Model (V-JEPA style)
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
        # V-JEPA-style masked denoising + stage-wise freezing
        "world_model.mask_ratio": "WM_MASK_RATIO",
        "world_model.mask_coef": "WM_MASK_COEF",
        "world_model.freeze_encoder_after": "WM_FREEZE_ENCODER_AFTER",
        # Optional anti-collapse regularization
        "world_model.variance_coef": "WM_VAR_COEF",
        "world_model.variance_target": "WM_VAR_TARGET",
        # Hybrid-action exploration helpers (delta-mode)
        "world_model.trigger_prob": "WM_TRIGGER_PROB",
        # Cursor instrumentation for click-dependent dynamics
        "world_model.mark_cursor": "WM_MARK_CURSOR",
        "world_model.cursor_token": "WM_CURSOR_TOKEN",
        "world_model.mark_prev_cursor": "WM_MARK_PREV_CURSOR",
        "world_model.prev_cursor_token": "WM_PREV_CURSOR_TOKEN",
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


def apply_ppo_config_if_present(
    *,
    override_env: bool = False,
    config_path: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """
    Convenience:
    - Reads TOML from PPO_CONFIG (or `ppo_config.toml` at repo root)
    - Applies it if the file exists
    """
    if config_path is None:
        config_path = os.environ.get("PPO_CONFIG", "ppo_config.toml")
    path = Path(config_path)
    if not path.is_file():
        return {}
    cfg = PPOConfig.load(path)
    return apply_ppo_config(cfg, override_env=override_env, env=env)


