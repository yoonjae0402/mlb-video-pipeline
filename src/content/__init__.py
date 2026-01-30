"""
MLB Video Pipeline - Content Package

AI-powered script generation for different video types:
- script_generator: Generate video scripts from game data
- prompts: Prompt templates for different content types
- middle_script: Series middle video scripts
- end_script: Series end video scripts
- explainer: Prediction reasoning generator

Usage:
    from src.content import ScriptGenerator, generate_middle_script

    generator = ScriptGenerator()
    script = generator.generate_game_recap(game_data)

    # Or use specific script generators
    from src.content import generate_middle_script, generate_end_script

    script = generate_middle_script(analysis, preview)
"""

from src.content.script_generator import ScriptGenerator
from src.content.prompts import PROMPTS
from src.content.middle_script import generate_middle_script
from src.content.end_script import generate_end_script
from src.content.explainer import PredictionExplainer

__all__ = [
    "ScriptGenerator",
    "PROMPTS",
    "generate_middle_script",
    "generate_end_script",
    "PredictionExplainer",
]
