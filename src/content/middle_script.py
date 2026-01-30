"""
MLB Video Pipeline - Series Middle Video Script Generator

Generates scripts for mid-series videos with format:
- [0-30s] Today's game analysis (post-game)
- [30-60s] Tomorrow's game preview with AI predictions

Total Duration: 60 seconds

Usage:
    from src.content.middle_script import generate_middle_script

    script = generate_middle_script(analysis_data, preview_data)
"""

from typing import Optional
import openai

from src.utils.logger import get_logger
from src.utils.cost_tracker import get_cost_tracker
from src.content.prompts import MIDDLE_ANALYSIS_PROMPT, MIDDLE_PREVIEW_PROMPT, get_system_prompt


logger = get_logger(__name__)
cost_tracker = get_cost_tracker()


def generate_middle_script(
    analysis_data: dict,
    preview_data: dict,
    model: str = "gpt-4o"
) -> str:
    """
    Generate script for series middle video.

    The script has two distinct parts:
    1. Analysis (30s): Recap of today's completed game
    2. Preview (30s): Tomorrow's matchup with AI predictions

    Args:
        analysis_data: Data from MLBStatsAnalyzer.analyze_game()
            {
                "game_result": {"home_team": "Dodgers", "away_team": "Yankees", ...},
                "key_insights": [...],
                "top_performances": [...],
                "storylines": [...]
            }
        preview_data: Data for tomorrow's game preview
            {
                "matchup": {"home": "Dodgers", "away": "Yankees"},
                "pitchers": {"home": {...}, "away": {...}},
                "predictions": [
                    {"player": "Shohei Ohtani", "prediction": "above_average",
                     "confidence": 0.78, "reasons": [...]}
                ]
            }
        model: OpenAI model to use

    Returns:
        Complete 60-second video script
    """
    logger.info("Generating series middle script")

    # Generate both parts
    analysis_script = _generate_analysis_section(analysis_data, model)
    preview_script = _generate_preview_section(preview_data, model)

    # Combine with transition
    full_script = _combine_scripts(analysis_script, preview_script)

    logger.info(f"Generated script: {len(full_script)} characters")
    return full_script


def _generate_analysis_section(analysis_data: dict, model: str) -> str:
    """
    Generate 30-second post-game analysis section.

    Covers:
    - Final score and result
    - 1-2 key highlights
    - Top performer with stats
    - Brief context (streak, standings impact)
    """
    # Format the prompt with analysis data
    game_result = analysis_data.get("game_result", {})
    insights = analysis_data.get("key_insights", [])
    performances = analysis_data.get("top_performances", [])
    storylines = analysis_data.get("storylines", [])

    prompt = f"""
{MIDDLE_ANALYSIS_PROMPT}

Game Result:
- {game_result.get('away_team')} {game_result.get('away_score')} at {game_result.get('home_team')} {game_result.get('home_score')}
- Venue: {game_result.get('venue', 'N/A')}

Key Insights:
{chr(10).join(f'- {insight}' for insight in insights[:3])}

Top Performance:
{_format_top_performance(performances[0] if performances else {})}

Storylines:
{chr(10).join(f'- {story}' for story in storylines[:2])}

Write a 30-second analysis script (approximately 75 words):
"""

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )

        script = response.choices[0].message.content.strip()

        # Track costs
        usage = response.usage
        cost_tracker.log_openai_call(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            model=model
        )

        return script

    except Exception as e:
        logger.error(f"Error generating analysis section: {e}")
        return _fallback_analysis(analysis_data)


def _generate_preview_section(preview_data: dict, model: str) -> str:
    """
    Generate 30-second preview section with AI predictions.

    Format:
    "AI predicts [player] will [performance level]
    First, [reason with stat]
    Second, [reason with stat]
    Third, [reason with stat]"
    """
    matchup = preview_data.get("matchup", {})
    pitchers = preview_data.get("pitchers", {})
    predictions = preview_data.get("predictions", [])

    # Format prediction reasons
    prediction_text = ""
    if predictions:
        pred = predictions[0]  # Focus on primary prediction
        reasons = pred.get("reasons", [])
        prediction_text = f"""
Player: {pred.get('player')}
Prediction: {pred.get('prediction')} (confidence: {pred.get('confidence', 0):.0%})
Reasons:
{chr(10).join(f'- {r.get("reason")}' for r in reasons[:3])}
"""

    prompt = f"""
{MIDDLE_PREVIEW_PROMPT}

Tomorrow's Matchup:
- {matchup.get('away', 'Away Team')} at {matchup.get('home', 'Home Team')}
- Starting Pitchers: {pitchers.get('away', {}).get('name', 'TBD')} vs {pitchers.get('home', {}).get('name', 'TBD')}

AI Prediction:
{prediction_text}

Write a 30-second preview script (approximately 75 words) that clearly explains WHY our AI made this prediction:
"""

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )

        script = response.choices[0].message.content.strip()

        # Track costs
        usage = response.usage
        cost_tracker.log_openai_call(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            model=model
        )

        return script

    except Exception as e:
        logger.error(f"Error generating preview section: {e}")
        return _fallback_preview(preview_data)


def _combine_scripts(analysis: str, preview: str) -> str:
    """Combine analysis and preview into full script with transition."""
    transition = "\n\n... Now, looking ahead to tomorrow's game ...\n\n"

    return f"{analysis}{transition}{preview}"


def _format_top_performance(performance: dict) -> str:
    """Format top performance for the prompt."""
    if not performance:
        return "No standout performance data available"

    player = performance.get("player", "Unknown")
    highlight = performance.get("highlight", "")
    stats = performance.get("stats", {})

    stat_line = ", ".join(
        f"{k}: {v}" for k, v in stats.items()
        if k in ["hits", "home_runs", "rbi", "runs"]
    )

    return f"- {player}: {highlight} ({stat_line})"


def _fallback_analysis(data: dict) -> str:
    """Fallback script when API fails."""
    game = data.get("game_result", {})
    return (
        f"The {game.get('home_team', 'home team')} faced the {game.get('away_team', 'visitors')} "
        f"in an exciting matchup. Final score: {game.get('away_score', 0)} to {game.get('home_score', 0)}. "
        f"Let's look at what made this game special."
    )


def _fallback_preview(data: dict) -> str:
    """Fallback script when API fails."""
    matchup = data.get("matchup", {})
    return (
        f"Tomorrow, the {matchup.get('away', 'visitors')} return to face the {matchup.get('home', 'home team')}. "
        f"Our AI model has analyzed the key matchups and has some interesting predictions to share."
    )
