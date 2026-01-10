"""
JEPA Agent UI - Pygame visualization for ARC-AGI-3.

This module provides the visual interface for the JEPA agent, including:
- Grid rendering with dynamic palette
- Cursor visualization
- Training metrics graphs
- Heatmap overlays
- Speed/dopamine/pain controls
"""

from .app import JEPAGameUI

def main():
    game = JEPAGameUI()
    game.run()

if __name__ == "__main__":
    main()
