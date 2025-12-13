import os
import time
import logging
from .agent import PPOAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CONFIGURATION
# 10 Games covering various skills
# Since we don't have real IDs, we'll use placeholders.
# If connecting to real API, replace these with valid IDs.
GAMES = [
    "game_target_click",   # 1. Target Click
    "game_drag",           # 2. Click & Drag
    "game_nav_arrows",     # 3. Arrow Keys
    "game_timing",         # 4. Timing
    "game_mixed",          # 5. Mixed
    "game_maze",           # 6. Avoidance
    "game_select",         # 7. Selection
    "game_memory",         # 8. Memory
    "game_reaction",       # 9. Reaction
    "game_idle",           # 10. Idle/Reset
]

# Architectures to sweep
ARCHITECTURES = {
    "A1_Baseline": {"use_attention": True, "use_lstm": True, "log_std_init": -0.5},
    "A2_NoAttention": {"use_attention": False, "use_lstm": True, "log_std_init": -0.5},
    "A3_NoLSTM": {"use_attention": True, "use_lstm": False, "log_std_init": -0.5},
    "A4_HighExploration": {"use_attention": True, "use_lstm": True, "log_std_init": 0.0},
}

MAX_STEPS_PER_RUN = 500 # Short run for testing sweep mechanism
USE_SYNTHETIC = True # Default to True for now since we lack credentials

def run_sweep():
    print("Starting Sweep...")
    
    results = []
    
    # Run full sweep
    test_games = GAMES
    test_archs = ARCHITECTURES
    
    for game_id in test_games:
        for arch_name, arch_config in test_archs.items():
            print(f"=== Running {game_id} with {arch_name} ===")
            
            # Create Agent
            # Note: PPOAgent init arguments (card_id, etc) are for the API.
            # We pass dummies if using synthetic.
            try:
                agent = PPOAgent(
                    card_id="dummy_card", 
                    game_id=game_id, 
                    agent_name=f"ppo_{arch_name}", 
                    ROOT_URL="http://localhost:8000", # Dummy
                    record=False,
                    use_gui=False
                )
                
                # Run Training
                start_time = time.time()
                agent.main(
                    max_steps=MAX_STEPS_PER_RUN, 
                    arch_config=arch_config, 
                    experiment_name=arch_name,
                    use_synthetic=USE_SYNTHETIC
                )
                duration = time.time() - start_time
                
                logger.info(f"Finished {game_id} / {arch_name} in {duration:.2f}s")
                res = {
                    "game": game_id, 
                    "arch": arch_name, 
                    "status": "success", 
                    "duration": duration
                }
                results.append(res)
                
                # Incremental Save
                import json
                try:
                    with open("sweep_results.json", "w") as f:
                        json.dump(results, f, indent=2)
                except: pass
                
                # Clean up GUI process to avoid zombie windows
                if agent.gui_process:
                    agent.cleanup()
                
            except Exception as e:
                logger.error(f"Failed run {game_id} / {arch_name}: {e}")
                results.append({
                    "game": game_id, 
                    "arch": arch_name, 
                    "status": "failed", 
                    "error": str(e)
                })
                
    # Summary
    logger.info("=== Sweep Summary ===")
    import json
    with open("sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    for res in results:
        logger.info(res)

if __name__ == "__main__":
    run_sweep()

