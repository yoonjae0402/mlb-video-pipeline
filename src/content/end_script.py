"""
MLB Video Pipeline - Series End Video Script Generator

Generates scripts for series-ending videos with format:
- [0-20s] Today's game + series summary
- [20-40s] Team 1's next opponent preview
- [40-60s] Team 2's next opponent preview

Total Duration: 60 seconds (compressed format)

Usage:
    from src.content.end_script import generate_end_script

    script = generate_end_script(analysis_data, series_summary, next_games)
"""

from typing import Optional
import openai

from src.utils.logger import get_logger
from src.utils.cost_tracker import get_cost_tracker
from src.content.prompts import END_SERIES_SUMMARY_PROMPT, END_NEXT_GAME_PROMPT, get_system_prompt


logger = get_logger(__name__)
cost_tracker = get_cost_tracker()


def generate_end_script(
    analysis_data: dict,
    series_summary: dict,
    next_games: dict,
    model: str = "gpt-4o"
) -> str:
    """
    Generate script for series end video.

    The script has three compressed parts:
    1. Series Summary (20s): Today's game + series result
    2. Team 1 Next (20s): First team's next opponent preview
    3. Team 2 Next (20s): Second team's next opponent preview

    Args:
        analysis_data: Data from MLBStatsAnalyzer.analyze_game()
        series_summary: Data from SeriesTracker.get_series_summary()
            {
                "series_score": "2-1",
                "series_winner": "Dodgers",
                "series_mvp": {"player": "...", "stats": "..."},
                "games": [...]
            }
        next_games: Data from SeriesTracker.get_next_opponents()
            {
                "Dodgers": {"opponent": "Padres", "game_date": "...", ...},
                "Yankees": {"opponent": "Red Sox", "game_date": "...", ...}
            }
        model: OpenAI model to use

    Returns:
        Complete 60-second video script
    """
    logger.info("Generating series end script")

    # Generate all three parts
    summary_script = _generate_series_summary(analysis_data, series_summary, model)

    # Get team names from series
    teams = list(next_games.keys())
    team1_preview = _generate_next_game_preview(teams[0], next_games.get(teams[0]), model) if len(teams) > 0 else ""
    team2_preview = _generate_next_game_preview(teams[1], next_games.get(teams[1]), model) if len(teams) > 1 else ""

    # Combine with transitions
    full_script = _combine_end_scripts(summary_script, team1_preview, team2_preview, teams)

    logger.info(f"Generated end script: {len(full_script)} characters")
    return full_script


def _generate_series_summary(
    analysis_data: dict,
    series_summary: dict,
    model: str
) -> str:
    """
    Generate 20-second series wrap-up section.

    Covers:
    - Today's game result
    - Series result (2-1, etc.)
    - Series MVP
    - One key takeaway
    """
    game_result = analysis_data.get("game_result", {})

    prompt = f"""
{END_SERIES_SUMMARY_PROMPT}

Today's Game:
- {game_result.get('away_team')} {game_result.get('away_score')} at {game_result.get('home_team')} {game_result.get('home_score')}

Series Result:
- Final: {series_summary.get('series_winner')} wins the series {series_summary.get('series_score')}
- Series MVP: {series_summary.get('series_mvp', {}).get('player', 'N/A')} - {series_summary.get('series_mvp', {}).get('stats', '')}

Write a 20-second series wrap-up (approximately 50 words):
"""

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
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
        logger.error(f"Error generating series summary: {e}")
        return _fallback_summary(analysis_data, series_summary)


def _generate_next_game_preview(
    team: str,
    next_game: Optional[dict],
    model: str
) -> str:
    """
    Generate 20-second next opponent preview.

    Covers:
    - Next opponent
    - Key matchup (pitcher or player)
    - Brief AI prediction with 1-2 reasons
    """
    if not next_game:
        return f"The {team} have a well-deserved day off before their next series."

    prompt = f"""
{END_NEXT_GAME_PROMPT}

Team: {team}
Next Opponent: {next_game.get('opponent')}
Game Date: {next_game.get('game_date')}
Location: {"Home" if next_game.get('is_home') else "Away"} at {next_game.get('venue', 'TBD')}
Probable Pitcher: {next_game.get('probable_pitcher', {}).get('name', 'TBD')}

Write a 20-second preview (approximately 50 words):
"""

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
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
        logger.error(f"Error generating next game preview for {team}: {e}")
        return _fallback_next_preview(team, next_game)


def _combine_end_scripts(
    summary: str,
    team1_preview: str,
    team2_preview: str,
    teams: list[str]
) -> str:
    """Combine all sections into full script with transitions."""
    parts = [summary]

    if team1_preview and len(teams) > 0:
        parts.append(f"\n\n... Now, what's next for the {teams[0]}? ...\n\n")
        parts.append(team1_preview)

    if team2_preview and len(teams) > 1:
        parts.append(f"\n\n... And for the {teams[1]}... ...\n\n")
        parts.append(team2_preview)

    return "".join(parts)


def _fallback_summary(analysis_data: dict, series_summary: dict) -> str:
    """Fallback script when API fails."""
    winner = series_summary.get('series_winner', 'The winning team')
    score = series_summary.get('series_score', '2-1')
    return f"{winner} takes the series {score}. A hard-fought battle comes to an end."


def _fallback_next_preview(team: str, next_game: Optional[dict]) -> str:
    """Fallback script when API fails."""
    if not next_game:
        return f"The {team} will look to rest up before their next matchup."

    opponent = next_game.get('opponent', 'their next opponent')
    return f"The {team} will face the {opponent} in their next series. An intriguing matchup awaits."


def generate_end_script_single_call(
    analysis_data: dict,
    series_summary: dict,
    next_games: dict,
    model: str = "gpt-4o"
) -> str:
    """
    Alternative: Generate entire end script in a single API call.

    More cost-efficient but less control over individual sections.
    """
    teams = list(next_games.keys())
    game_result = analysis_data.get("game_result", {})

    prompt = f"""
Create a 60-second video script for a series-ending MLB video with three parts:

PART 1 - Series Wrap-up (20 seconds):
Today's Game: {game_result.get('away_team')} {game_result.get('away_score')} at {game_result.get('home_team')} {game_result.get('home_score')}
Series Winner: {series_summary.get('series_winner')} ({series_summary.get('series_score')})
Series MVP: {series_summary.get('series_mvp', {}).get('player', 'N/A')}

PART 2 - {teams[0] if teams else 'Team 1'}'s Next Game (20 seconds):
{_format_next_game(teams[0], next_games.get(teams[0])) if teams else 'N/A'}

PART 3 - {teams[1] if len(teams) > 1 else 'Team 2'}'s Next Game (20 seconds):
{_format_next_game(teams[1], next_games.get(teams[1])) if len(teams) > 1 else 'N/A'}

Requirements:
- Keep each section to approximately 50 words
- Use natural transitions between sections
- Include brief AI predictions where relevant
- Make it engaging for both teams' fans

Write the complete script:
"""

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
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
        logger.error(f"Error generating end script: {e}")
        # Fall back to multi-call approach
        return generate_end_script(analysis_data, series_summary, next_games, model)


def _format_next_game(team: str, next_game: Optional[dict]) -> str:
    """Format next game info for prompt."""
    if not next_game:
        return f"{team}: Day off"

    return f"""
Opponent: {next_game.get('opponent')}
Date: {next_game.get('game_date')}
Location: {"Home" if next_game.get('is_home') else "Away"}
"""
