"""Entry point for the JEPA UI subprocess."""
import sys
import os

# Help pygame work properly on Windows without a visible console
if sys.platform == "win32":
    # Ensure pygame can find a display
    os.environ.setdefault("SDL_VIDEODRIVER", "windows")
    # Ensure pygame audio doesn't hang
    os.environ.setdefault("SDL_AUDIODRIVER", "directsound")

# Ensure the package can be found
from .app import JEPAGameUI

def main():
    try:
        print(f"LOADING JEPA UI FROM: {__file__}")
        sys.stdout.flush()
        game = JEPAGameUI()
        game.run()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()

if __name__ == "__main__":
    main()
