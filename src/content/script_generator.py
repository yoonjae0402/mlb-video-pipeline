"""
MLB Video Pipeline - Script Generator

Uses OpenAI GPT-4 to generate engaging video scripts from game data.
Handles prompt construction, API calls, and response processing.

Usage:
    from src.content.script_generator import ScriptGenerator

    generator = ScriptGenerator()

    # Generate a game recap script
    script = generator.generate_game_recap(game_data)

    # Generate a prediction script
    script = generator.generate_prediction(matchup_data)
"""

from pathlib import Path
from typing import Any
from datetime import datetime

from openai import OpenAI

from config.settings import settings
from config.league_config import MLB_TEAMS
from src.content.prompts import (
    format_prompt,
    get_system_prompt,
    apply_modifiers,
    estimate_duration,
)
from src.utils.logger import get_logger
from src.utils.cost_tracker import get_cost_tracker
from src.utils.validators import validate_script


logger = get_logger(__name__)


class ScriptGenerator:
    """
    Generate video scripts using GPT-4.

    Provides methods for different content types with automatic
    prompt construction and cost tracking.
    """

    # OpenAI pricing per 1K tokens (as of 2025-01)
    # See: https://openai.com/api/pricing/
    PRICING = {
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        output_dir: Path | None = None,
    ):
        """
        Initialize the script generator.

        Args:
            api_key: OpenAI API key (uses settings if not provided)
            model: GPT model to use
            output_dir: Directory to save generated scripts
        """
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")

        self.model = model
        self.output_dir = output_dir or settings.scripts_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.client = OpenAI(api_key=self.api_key)
        self.cost_tracker = get_cost_tracker()

        logger.info(f"ScriptGenerator initialized with model: {model}")

    def _call_gpt(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> tuple[str, dict]:
        """
        Make a GPT API call.

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens in response
            temperature: Creativity setting (0-1)

        Returns:
            Tuple of (generated_text, usage_info)
        """
        max_tokens = max_tokens or settings.openai_max_tokens

        if settings.dry_run:
            logger.info("DRY RUN: Would call GPT API")
            return "[DRY RUN] Script would be generated here.", {"total_tokens": 0}

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": get_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Extract response
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            # Calculate and track cost
            cost = self._calculate_cost(usage)
            self.cost_tracker.log_api_call("openai", cost, tokens=usage["total_tokens"])

            logger.info(f"GPT call completed: {usage['total_tokens']} tokens, ${cost:.4f}")

            return content, usage

        except Exception as e:
            logger.error(f"GPT API call failed: {e}")
            raise

    def _calculate_cost(self, usage: dict) -> float:
        """Calculate cost from token usage."""
        pricing = self.PRICING.get(self.model, self.PRICING["gpt-4-turbo"])
        input_cost = (usage["prompt_tokens"] / 1000) * pricing["input"]
        output_cost = (usage["completion_tokens"] / 1000) * pricing["output"]
        return input_cost + output_cost

    def _format_stats(self, stats: dict[str, Any]) -> str:
        """Format statistics dictionary for prompt."""
        lines = []
        for key, value in stats.items():
            # Clean up key names
            display_key = key.replace("_", " ").title()
            lines.append(f"- {display_key}: {value}")
        return "\n".join(lines)

    def _save_script(self, script: str, script_type: str, identifier: str) -> Path:
        """Save generated script to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{script_type}_{identifier}_{timestamp}.txt"
        filepath = self.output_dir / filename

        filepath.write_text(script)
        logger.info(f"Script saved to {filepath}")

        return filepath

    # =========================================================================
    # Content Generation Methods
    # =========================================================================

    def generate_game_recap(
        self,
        game_data: dict[str, Any],
        duration: int = 60,
        tone: str = "exciting",
        save: bool = True,
    ) -> str:
        """
        Generate a game recap script.

        Args:
            game_data: Game information dictionary
            duration: Target video duration in seconds
            tone: Script tone (exciting, analytical, casual)
            save: Whether to save the script to file

        Returns:
            Generated script text
        """
        logger.info(f"Generating game recap for game {game_data.get('game_id')}")

        # Build prompt data
        home_team = MLB_TEAMS.get(game_data.get("home_team_id"), {})
        away_team = MLB_TEAMS.get(game_data.get("away_team_id"), {})

        prompt = format_prompt(
            "game_recap",
            duration=duration,
            home_team=home_team.get("name", "Home Team"),
            away_team=away_team.get("name", "Away Team"),
            home_record=game_data.get("home_record", "0-0"),
            away_record=game_data.get("away_record", "0-0"),
            home_score=game_data.get("home_score", 0),
            away_score=game_data.get("away_score", 0),
            game_date=game_data.get("game_date", ""),
            venue=game_data.get("venue", ""),
            key_stats=self._format_stats(game_data.get("key_stats", {})),
            notable_performances=game_data.get("notable_performances", "None recorded"),
        )

        # Apply tone modifier
        prompt = apply_modifiers(prompt, tone=tone)

        # Generate script
        script, usage = self._call_gpt(prompt, temperature=0.7)

        # Validate
        is_valid, error = validate_script(script)
        if not is_valid:
            logger.warning(f"Script validation warning: {error}")

        # Save if requested
        if save:
            self._save_script(script, "game_recap", str(game_data.get("game_id", "unknown")))

        return script

    def generate_player_spotlight(
        self,
        player_data: dict[str, Any],
        duration: int = 45,
        save: bool = True,
    ) -> str:
        """
        Generate a player spotlight script.

        Args:
            player_data: Player information and statistics
            duration: Target video duration in seconds
            save: Whether to save the script

        Returns:
            Generated script text
        """
        logger.info(f"Generating spotlight for player {player_data.get('player_name')}")

        prompt = format_prompt(
            "player_spotlight",
            duration=duration,
            player_name=player_data.get("player_name", "Unknown Player"),
            team_name=player_data.get("team_name", "Unknown Team"),
            position=player_data.get("position", "Unknown"),
            time_period=player_data.get("time_period", "Last 7 days"),
            recent_stats=self._format_stats(player_data.get("recent_stats", {})),
            season_stats=self._format_stats(player_data.get("season_stats", {})),
            context=player_data.get("context", ""),
        )

        script, usage = self._call_gpt(prompt, temperature=0.6)

        if save:
            player_id = player_data.get("player_id", "unknown")
            self._save_script(script, "player_spotlight", str(player_id))

        return script

    def generate_prediction(
        self,
        matchup_data: dict[str, Any],
        model_predictions: dict[str, Any],
        duration: int = 60,
        save: bool = True,
    ) -> str:
        """
        Generate a game prediction script.

        Args:
            matchup_data: Game matchup information
            model_predictions: ML model prediction results
            duration: Target video duration
            save: Whether to save the script

        Returns:
            Generated script text
        """
        logger.info(f"Generating prediction for matchup")

        home_team = MLB_TEAMS.get(matchup_data.get("home_team_id"), {})
        away_team = MLB_TEAMS.get(matchup_data.get("away_team_id"), {})

        # Format model predictions
        pred_text = []
        for key, value in model_predictions.items():
            if isinstance(value, float):
                pred_text.append(f"- {key}: {value:.1%}")
            else:
                pred_text.append(f"- {key}: {value}")

        prompt = format_prompt(
            "prediction",
            duration=duration,
            home_team=home_team.get("name", "Home Team"),
            away_team=away_team.get("name", "Away Team"),
            game_date=matchup_data.get("game_date", ""),
            home_pitcher=matchup_data.get("home_pitcher", "TBD"),
            away_pitcher=matchup_data.get("away_pitcher", "TBD"),
            team_stats=self._format_stats(matchup_data.get("team_stats", {})),
            head_to_head=matchup_data.get("head_to_head", "No recent history"),
            model_predictions="\n".join(pred_text),
            factors=matchup_data.get("factors", ""),
        )

        script, usage = self._call_gpt(prompt, temperature=0.7)

        if save:
            game_id = matchup_data.get("game_id", "unknown")
            self._save_script(script, "prediction", str(game_id))

        return script

    def generate_weekly_roundup(
        self,
        week_data: dict[str, Any],
        duration: int = 120,
        save: bool = True,
    ) -> str:
        """
        Generate a weekly roundup script.

        Args:
            week_data: Week's games and highlights
            duration: Target video duration
            save: Whether to save the script

        Returns:
            Generated script text
        """
        logger.info("Generating weekly roundup")

        prompt = format_prompt(
            "weekly_roundup",
            duration=duration,
            week_start=week_data.get("week_start", ""),
            week_end=week_data.get("week_end", ""),
            standings_changes=week_data.get("standings_changes", ""),
            top_performances=week_data.get("top_performances", ""),
            storylines=week_data.get("storylines", ""),
            upcoming=week_data.get("upcoming", ""),
        )

        script, usage = self._call_gpt(prompt, temperature=0.7)

        if save:
            self._save_script(script, "weekly_roundup", week_data.get("week_start", "unknown"))

        return script

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_estimated_duration(self, script: str) -> int:
        """Get estimated narration duration for a script."""
        return estimate_duration(script)

    def get_cost_summary(self) -> dict[str, Any]:
        """Get cost tracking summary."""
        return self.cost_tracker.get_summary()

    def check_budget(self) -> bool:
        """Check if within daily budget."""
        return self.cost_tracker.check_budget(settings.daily_cost_limit)
