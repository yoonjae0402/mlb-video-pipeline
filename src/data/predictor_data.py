"""
MLB Video Pipeline - Prediction Data Processor

Handles data preparation for the prediction model, including
cross-season data handling for early-season predictions.

Key Feature: Cross-season data handling
- If current season has < 10 games, pull from previous season
- Adds season_break feature to indicate off-season gap

Usage:
    from src.data.predictor_data import PredictionDataProcessor

    processor = PredictionDataProcessor()
    sequence = processor.get_player_sequence("Shohei Ohtani", "2024-04-05")
"""

from typing import Optional, Literal
from datetime import datetime, timedelta
import numpy as np

from src.utils.logger import get_logger
from src.data.fetcher import MLBDataFetcher
from config.league_config import CROSS_SEASON_CONFIG


logger = get_logger(__name__)


# Feature indices for the 11-feature vector
FEATURE_NAMES = [
    "batting_average",      # 0: 10-game rolling BA
    "on_base_pct",          # 1: 10-game rolling OBP
    "slugging_pct",         # 2: 10-game rolling SLG
    "home_runs",            # 3: HR in this game
    "rbi",                  # 4: RBI in this game
    "hits",                 # 5: Hits in this game
    "at_bats",              # 6: At bats in this game
    "walks",                # 7: Walks in this game
    "strikeouts",           # 8: Strikeouts in this game
    "is_home",              # 9: 1 if home game, 0 if away
    "season_break",         # 10: 1 if gap > 120 days (off-season)
]


class PredictionDataProcessor:
    """
    Prepare features for the prediction model.

    Handles:
    - Feature extraction from game logs
    - Cross-season data merging
    - Edge cases (rookies, injuries, trades)
    - Sequence padding and normalization
    """

    def __init__(
        self,
        fetcher: Optional[MLBDataFetcher] = None,
        sequence_length: int = None,
    ):
        """
        Initialize the data processor.

        Args:
            fetcher: MLBDataFetcher instance for API calls
            sequence_length: Number of games in sequence (default from config)
        """
        self.fetcher = fetcher or MLBDataFetcher()
        self.sequence_length = sequence_length or CROSS_SEASON_CONFIG.get("sequence_length", 10)
        self.max_lookback_seasons = CROSS_SEASON_CONFIG.get("max_lookback_seasons", 2)
        self.min_games_for_prediction = CROSS_SEASON_CONFIG.get("min_games_for_prediction", 5)

    def get_player_sequence(
        self,
        player: str,
        current_date: str,
        sequence_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Get the last N games for a player, crossing seasons if needed.

        Example for April 5, 2024:
        - Current season: 5 games played
        - Previous season: Pull last 5 games of 2023
        - Total: 10 games with season_break=1 at the boundary

        Args:
            player: Player name or ID
            current_date: Date string (YYYY-MM-DD) for the prediction
            sequence_length: Override default sequence length

        Returns:
            np.ndarray: Shape (sequence_length, 11)
                       11 features per game including season_break flag
        """
        seq_len = sequence_length or self.sequence_length
        logger.info(f"Getting {seq_len}-game sequence for {player} before {current_date}")

        # Parse date
        pred_date = datetime.strptime(current_date, "%Y-%m-%d")
        current_year = pred_date.year

        # Get current season games
        current_games = self._get_season_games(player, current_year, before_date=current_date)
        logger.debug(f"Found {len(current_games)} games in current season")

        # Check if we need previous season data
        games_needed = seq_len - len(current_games)

        if games_needed > 0 and len(current_games) < seq_len:
            logger.info(f"Need {games_needed} more games, checking previous season")

            # Get previous season games
            prev_games = self._get_season_games(player, current_year - 1)

            if prev_games:
                # Take the last N games from previous season
                prev_games = prev_games[-games_needed:]

                # Mark the boundary with season_break
                if prev_games:
                    prev_games[-1]["season_break"] = 1

                # Combine: previous season + current season
                all_games = prev_games + current_games
                logger.info(f"Combined {len(prev_games)} prev + {len(current_games)} current games")
            else:
                all_games = current_games
        else:
            all_games = current_games[-seq_len:]

        # Convert to feature matrix
        features = self._games_to_features(all_games, seq_len)

        return features

    def _get_season_games(
        self,
        player: str,
        season: int,
        before_date: Optional[str] = None
    ) -> list[dict]:
        """
        Get all games for a player in a specific season.

        Args:
            player: Player name or ID
            season: Year (e.g., 2024)
            before_date: Only include games before this date

        Returns:
            List of game dictionaries with stats
        """
        try:
            # This would call the fetcher to get game logs
            # For now, return placeholder
            games = []

            # TODO: Implement actual API call
            # games = self.fetcher.get_player_game_logs(player, season)

            # Filter by date if specified
            if before_date and games:
                before = datetime.strptime(before_date, "%Y-%m-%d")
                games = [g for g in games
                        if datetime.strptime(g.get("date"), "%Y-%m-%d") < before]

            return games

        except Exception as e:
            logger.error(f"Error fetching games for {player} in {season}: {e}")
            return []

    def _games_to_features(self, games: list[dict], target_length: int) -> np.ndarray:
        """
        Convert game logs to feature matrix.

        Args:
            games: List of game dictionaries
            target_length: Desired sequence length

        Returns:
            np.ndarray: Shape (target_length, 11)
        """
        num_features = len(FEATURE_NAMES)
        features = np.zeros((target_length, num_features))

        for i, game in enumerate(games[-target_length:]):
            idx = target_length - len(games) + i
            if idx >= 0:
                features[idx] = self._extract_game_features(game)

        return features

    def _extract_game_features(self, game: dict) -> np.ndarray:
        """Extract 11 features from a single game."""
        return np.array([
            game.get("batting_average", 0.0),
            game.get("on_base_pct", 0.0),
            game.get("slugging_pct", 0.0),
            game.get("home_runs", 0),
            game.get("rbi", 0),
            game.get("hits", 0),
            game.get("at_bats", 0),
            game.get("walks", 0),
            game.get("strikeouts", 0),
            1.0 if game.get("is_home", False) else 0.0,
            1.0 if game.get("season_break", 0) else 0.0,
        ], dtype=np.float32)

    def handle_insufficient_data(
        self,
        player: str,
        available_games: int
    ) -> Literal["use_available", "skip_prediction", "use_minors"]:
        """
        Handle edge cases when player has insufficient data.

        Cases:
        - Rookie with no MLB history → check minors or skip
        - Player returning from long injury → use available
        - Recently traded player → use available (all leagues)

        Args:
            player: Player name or ID
            available_games: Number of games found

        Returns:
            Strategy: "use_available" | "skip_prediction" | "use_minors"
        """
        if available_games >= self.min_games_for_prediction:
            return "use_available"

        # Check if player is a rookie (no previous MLB seasons)
        is_rookie = self._is_rookie(player)

        if is_rookie:
            if CROSS_SEASON_CONFIG.get("include_minors", False):
                return "use_minors"
            else:
                return "skip_prediction"

        # Player has some history, use what's available
        if available_games > 0:
            return "use_available"

        return "skip_prediction"

    def _is_rookie(self, player: str) -> bool:
        """Check if player is a rookie (no previous MLB seasons)."""
        # TODO: Implement with actual API
        return False

    def get_matchup_features(
        self,
        batter: str,
        pitcher: str,
        game_date: str
    ) -> dict:
        """
        Get batter vs pitcher matchup features.

        Useful for prediction explanations.

        Args:
            batter: Batter name/ID
            pitcher: Pitcher name/ID
            game_date: Date of the matchup

        Returns:
            Dictionary with matchup statistics
        """
        return {
            "at_bats": 0,
            "hits": 0,
            "home_runs": 0,
            "strikeouts": 0,
            "batting_average": 0.0,
            "sample_size": "insufficient",  # or "small", "medium", "large"
        }

    def get_split_features(
        self,
        player: str,
        split_type: str,
        season: int
    ) -> dict:
        """
        Get player splits (home/away, vs L/R, day/night).

        Args:
            player: Player name/ID
            split_type: "home_away" | "platoon" | "day_night"
            season: Season year

        Returns:
            Dictionary with split statistics
        """
        return {
            "split_type": split_type,
            "primary": {},
            "secondary": {},
            "difference": {},
        }

    def prepare_batch_input(
        self,
        players: list[str],
        game_date: str
    ) -> np.ndarray:
        """
        Prepare input batch for multiple players.

        Args:
            players: List of player names/IDs
            game_date: Prediction date

        Returns:
            np.ndarray: Shape (num_players, sequence_length, 11)
        """
        batch = []

        for player in players:
            try:
                sequence = self.get_player_sequence(player, game_date)
                batch.append(sequence)
            except Exception as e:
                logger.warning(f"Could not get data for {player}: {e}")
                # Append zeros for failed players
                batch.append(np.zeros((self.sequence_length, len(FEATURE_NAMES))))

        return np.array(batch)
