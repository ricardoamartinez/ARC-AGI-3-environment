import json
import logging
import os
import time
import queue
import threading

logger = logging.getLogger()


class LiveVisualizerCallback:
    """
    Minimal live visualization: show grids, cursor, last action, and feedback reward.
    No saliency/heatmap extraction or other heuristic maps.
    """

    def __init__(self, gui_process, agent):
        self.gui_process = gui_process
        self.agent = agent
        self._quit_event = None
        self.step_count = 0

        # UI should never lag behind training: keep only the latest message (coalescing queue).
        self.msg_queue = queue.Queue(maxsize=1)
        self._last_send_ts = 0.0
        self._ui_fps = float(os.environ.get("JEPA_UI_FPS", "60"))
        self._ui_fps = max(1.0, min(240.0, self._ui_fps))
        # Sending full grids is expensive (huge JSON). Send grids at a lower FPS, and only when needed.
        self._grid_fps = float(os.environ.get("JEPA_UI_GRID_FPS", "5"))
        self._grid_fps = max(0.1, min(60.0, self._grid_fps))
        self._last_grid_send_ts = 0.0
        self._last_grids_obj_id = None
        self._writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._writer_thread.start()

    def _write_loop(self):
        while True:
            msg = self.msg_queue.get()
            if msg is None:
                return
            if not self.gui_process or self.gui_process.poll() is not None:
                continue
            try:
                self.gui_process.stdin.write(json.dumps(msg) + "\n")
                self.gui_process.stdin.flush()
            except (BrokenPipeError, IOError):
                return

    def on_step(self, info: dict = None) -> bool:
        self.step_count += 1

        if self._quit_event and self._quit_event.is_set():
            logger.info("Quit signal detected in callback, stopping training...")
            return False

        # Fast path: check throttle FIRST before any other work
        now = time.time()
        min_dt = 1.0 / self._ui_fps
        if now - self._last_send_ts < min_dt:
            return True  # Skip this step - nothing else to do
        self._last_send_ts = now

        # Only do work if we're actually going to send an update
        latest_frame = self.agent.frames[-1] if self.agent.frames else None
        grids = None
        score = 0
        if latest_frame and latest_frame.frame:
            grids = latest_frame.frame
            score = int(getattr(latest_frame, "score", 0))
        else:
            # Allow cursor-only training without server frames.
            grids = getattr(self.agent, "_latest_grid_for_ui", None)
            if not grids:
                return True
        
        # Decide whether to include the (large) grids payload this tick.
        include_grids = False
        gid = id(grids)
        if self._last_grids_obj_id is None or gid != self._last_grids_obj_id:
            include_grids = True
        else:
            min_grid_dt = 1.0 / self._grid_fps
            if now - self._last_grid_send_ts >= min_grid_dt:
                include_grids = True
        if include_grids:
            self._last_grids_obj_id = gid
            self._last_grid_send_ts = now

        last_action = getattr(self.agent, "_last_action_viz", None)
        reward = float(info.get("reward", 0.0)) if info else 0.0
        manual_dopamine = float(info.get("manual_dopamine", 0.0)) if info else 0.0
        manual_pain = float(info.get("manual_pain", 0.0)) if info else 0.0
        goal = info.get("goal") if info else None
        goal_dist = info.get("goal_dist") if info else None
        goal_progress = info.get("goal_progress") if info else None
        goal_shaping_enabled = info.get("goal_shaping_enabled") if info else None
        goal_version = info.get("goal_version") if info else None
        grid_min = info.get("grid_min") if info else None
        grid_max = info.get("grid_max") if info else None
        cursor_speed = info.get("cursor_speed") if info else None
        action_energy = info.get("action_energy") if info else None
        energy_penalty_coef = info.get("energy_penalty_coef") if info else None

        msg = {
            "game_id": self.agent.game_id,
            "score": score,
            "state": f"Step: {self.step_count}",
            "last_action": last_action,
            "cursor": {"x": self.agent.cursor_x, "y": self.agent.cursor_y},
            "goal": goal,
            "metrics": {
                # Provide both legacy and canonical keys for the UI.
                "reward": reward,
                "reward_mean": reward,
                "manual_dopamine": manual_dopamine,
                "manual_pain": manual_pain,
                "trigger": float(info.get("trigger", 0.0)) if info else 0.0,
                "goal_dist": float(goal_dist) if goal_dist is not None else None,
                "goal_progress": float(goal_progress) if goal_progress is not None else None,
                "goal_shaping_enabled": goal_shaping_enabled,
                "goal_version": goal_version,
                "grid_min": grid_min,
                "grid_max": grid_max,
                "cursor_speed": cursor_speed,
                "action_energy": action_energy,
                "energy_penalty_coef": energy_penalty_coef,
            },
        }
        if include_grids:
            msg["grids"] = grids

        # Coalesce: if queue is full, drop the old message and replace with the newest.
        try:
            if self.msg_queue.full():
                _ = self.msg_queue.get_nowait()
            self.msg_queue.put_nowait(msg)
        except queue.Full:
            pass
        return True


