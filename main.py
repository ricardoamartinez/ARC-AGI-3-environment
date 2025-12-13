# ruff: noqa: E402
import os
from dotenv import load_dotenv

# IMPORTANT:
# `.env.example` is a template and may contain experimental flags (like headless mode).
# Loading it at runtime can unintentionally force behavior (e.g., disabling the UI).
# So we load only `.env` by default. If you *really* want `.env.example`, set
# LOAD_ENV_EXAMPLE=1 in your environment.
if os.environ.get("LOAD_ENV_EXAMPLE", "0") == "1":
    load_dotenv(dotenv_path=".env.example")
load_dotenv(dotenv_path=".env", override=True)

import argparse
import json
import logging
import signal
import sys
import threading
import time
from functools import partial
from types import FrameType
from typing import Optional

import requests

from agents import AVAILABLE_AGENTS, Swarm
from agents.tracing import initialize as init_agentops
from agents.ppo.config import apply_ppo_config_if_present

logger = logging.getLogger()

# Default to the public ARC-AGI-3 server. You can override with SCHEME/HOST/PORT
# if you're running a local API.
SCHEME = os.environ.get("SCHEME", "https")
HOST = os.environ.get("HOST", "three.arcprize.org")
PORT = os.environ.get("PORT", 443)

# Hide standard ports in URL
if (SCHEME == "http" and str(PORT) == "80") or (
    SCHEME == "https" and str(PORT) == "443"
):
    ROOT_URL = f"{SCHEME}://{HOST}"
else:
    ROOT_URL = f"{SCHEME}://{HOST}:{PORT}"

# API Key: Use env var if valid, otherwise use provided default
_api_key = os.getenv("ARC_API_KEY", "")
if not _api_key or _api_key.startswith("your_"):
    _api_key = "894cb2d9-45a5-4897-91e8-f03d7d4a1f8a"

HEADERS = {
    "X-API-Key": _api_key,
    "Accept": "application/json",
}


def run_agent(swarm: Swarm) -> None:
    swarm.main()
    os.kill(os.getpid(), signal.SIGINT)


def cleanup(
    swarm: Swarm,
    signum: Optional[int],
    frame: Optional[FrameType],
) -> None:
    logger.info("Received SIGINT, exiting...")
    card_id = swarm.card_id
    if card_id:
        scorecard = swarm.close_scorecard(card_id)
        if scorecard:
            logger.info("--- EXISTING SCORECARD REPORT ---")
            logger.info(json.dumps(scorecard.model_dump(), indent=2))
            swarm.cleanup(scorecard)

        # Provide web link to scorecard
        if card_id:
            scorecard_url = f"{ROOT_URL}/scorecards/{card_id}"
            logger.info(f"View your scorecard online: {scorecard_url}")

    sys.exit(0)


def main() -> None:
    print("Starting main()...", flush=True)
    log_level = logging.INFO
    if os.environ.get("DEBUG", "False") == "True":
        log_level = logging.DEBUG

    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)

    # NOTE: multiple concurrent runs can corrupt a single shared log file.
    # Write each run to its own log file.
    os.makedirs("logs", exist_ok=True)
    run_id = int(time.time())
    file_handler = logging.FileHandler(os.path.join("logs", f"run-{run_id}.log"), mode="w")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    # logging.getLogger("requests").setLevel(logging.CRITICAL)
    # logging.getLogger("werkzeug").setLevel(logging.CRITICAL)

    parser = argparse.ArgumentParser(description="ARC-AGI-3-Agents")
    parser.add_argument(
        "-a",
        "--agent",
        choices=AVAILABLE_AGENTS.keys(),
        help="Choose which agent to run.",
    )
    parser.add_argument(
        "-g",
        "--game",
        help="Choose a specific game_id for the agent to play. If none specified, an agent swarm will play all available games.",
    )
    parser.add_argument(
        "-t",
        "--tags",
        type=str,
        help="Comma-separated list of tags for the scorecard (e.g., 'experiment,v1.0')",
        default=None,
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run PPOAgent headless (no pygame UI). By default PPOAgent opens the UI.",
    )

    args = parser.parse_args()

    if not args.agent:
        logger.error("An Agent must be specified")
        return

    # Apply PPO config file (if present) after dotenv but before agent construction.
    # This lets you keep all knobs in one file (`ppo_config.toml`) instead of many env vars.
    applied_cfg = apply_ppo_config_if_present(override_env=False)
    if applied_cfg:
        logger.info("Applied ppo_config.toml: %s", applied_cfg)

    # PPO UI/headless handling:
    # Many users set PPO_NO_GUI/PPO_RANDOM_GOAL for headless tests and forget to unset it.
    # We default to UI when running ppoagent unless --no-gui is explicitly passed.
    if args.agent == "ppoagent":
        if getattr(args, "no_gui", False):
            os.environ["PPO_NO_GUI"] = "1"
            logger.info("PPOAgent will run headless (--no-gui).")
        else:
            if "PPO_NO_GUI" in os.environ:
                os.environ.pop("PPO_NO_GUI", None)
            # Also clear headless-only goal envs so UI runs rely on clicking the goal.
            os.environ.pop("PPO_RANDOM_GOAL", None)
            os.environ.pop("PPO_FIXED_GOAL", None)
            logger.info("PPOAgent will run with UI (default).")

    print(f"{ROOT_URL}/api/games")
    print(f"Using API Key: {HEADERS['X-API-Key'][:8]}...{HEADERS['X-API-Key'][-4:]}")

    # Get the list of games from the API
    full_games = []
    try:
        with requests.Session() as session:
            session.headers.update(HEADERS)
            r = session.get(f"{ROOT_URL}/api/games", timeout=10)

        if r.status_code == 200:
            try:
                response_json = r.json()
                print(f"API returned {len(response_json)} games")
                full_games = [g["game_id"] for g in response_json]
            except (ValueError, KeyError) as e:
                logger.error(f"Failed to parse games response: {e}")
                logger.error(f"Response content: {r.text[:200]}")
        else:
            logger.error(
                f"API request failed with status {r.status_code}: {r.text[:200]}"
            )

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to API server: {e}")

    # For playback agents, we can derive the game from the recording filename
    if not full_games and args.agent and args.agent.endswith(".recording.jsonl"):
        from agents.recorder import Recorder

        game_prefix = Recorder.get_prefix_one(args.agent)
        full_games = [game_prefix]
        logger.info(
            f"Using game '{game_prefix}' derived from playback recording filename"
        )
    
    # If API returns empty but user provided a game ID, use it directly
    if not full_games and args.game:
        logger.info(
            f"API returned empty game list, but using provided game ID: {args.game}"
        )
        full_games = args.game.split(",")
    
    # INTERACTIVE SELECTION FOR PPO AGENT
    if args.agent == "ppoagent" and not args.game:
        from agents.ppo.agent import PPOAgent
        logger.info("No game specified for PPOAgent. Launching interactive game selector...")
        selected_id = PPOAgent.select_game_interactively(ROOT_URL, HEADERS)
        if selected_id:
            logger.info(f"Selected game: {selected_id}")
            # Treat as if the user passed this game ID
            args.game = selected_id
            full_games = [selected_id]
        else:
            logger.info("No game selected or selection cancelled. Exiting.")
            return

    games = full_games[:]
    if args.game:
        filters = args.game.split(",")
        games = [
            gid
            for gid in full_games
            if any(gid.startswith(prefix) for prefix in filters)
        ]

    logger.info(f"Game list: {games}")

    if not games:
        if full_games:
            logger.error(
                f"The specified game '{args.game}' does not exist or is not available with your API key. Please try a different game."
            )
        else:
            logger.error(
                "No games available to play. Check API connection or recording file."
            )
        return

    # Start with Empty tags, "agent" and agent name will be added by the Swarm later
    tags = []

    # Append user-provided tags if any
    if args.tags:
        user_tags = [tag.strip() for tag in args.tags.split(",")]
        tags.extend(user_tags)

    # Initialize AgentOps client
    init_agentops(api_key=os.getenv("AGENTOPS_API_KEY"), log_level=log_level)

    swarm = Swarm(
        args.agent,
        ROOT_URL,
        games,
        tags=tags,  # Pass tags as keyword argument
    )
    agent_thread = threading.Thread(target=partial(run_agent, swarm))
    agent_thread.daemon = True  # die when the main thread dies
    agent_thread.start()

    signal.signal(signal.SIGINT, partial(cleanup, swarm))  # handler for Ctrl+C

    try:
        # Wait for the agent thread to complete
        while agent_thread.is_alive():
            agent_thread.join(timeout=5)  # Check every 5 second
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received in main thread")
        cleanup(swarm, signal.SIGINT, None)
    except Exception as e:
        logger.error(f"Unexpected error in main thread: {e}")
        cleanup(swarm, None, None)


if __name__ == "__main__":
    os.environ["TESTING"] = "False"
    main()
