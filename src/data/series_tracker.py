"""
MLB Video Pipeline - Series Tracking

Tracks series progress and determines video type:
- series_middle: Same opponents play again tomorrow
- series_end: Series is complete, different opponents next

Usage:
    from src.data.series_tracker import SeriesTracker

    tracker = SeriesTracker()
    video_type = tracker.get_video_type(game)  # "series_middle" or "series_end"
"""

from typing import Optional
from datetime import datetime, timedelta

from src.utils.logger import get_logger
from src.data.fetcher import MLBDataFetcher


logger = get_logger(__name__)


class SeriesTracker:
    """
    Track MLB series progress and determine video content type.

    MLB Series Rules:
    - Regular season: Usually 3-game series (some 2 or 4 game series)
    - Check tomorrow's schedule to determine if series continues
    - Series end triggers different video format
    """

    def __init__(self, fetcher: Optional[MLBDataFetcher] = None):
        """
        Initialize series tracker.

        Args:
            fetcher: MLBDataFetcher instance for schedule lookups
        """
        self.fetcher = fetcher or MLBDataFetcher()

    def get_video_type(self, game: dict) -> str:
        """
        Determine video type based on series position.

        Logic:
        1. Get tomorrow's schedule for both teams
        2. If same teams play each other → series_middle
        3. If different opponents → series_end

        Args:
            game: Game data dictionary with home_team, away_team, date

        Returns:
            "series_middle" | "series_end"
        """
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        game_date = game.get("date")

        if not all([home_team, away_team, game_date]):
            logger.warning("Incomplete game data, defaulting to series_middle")
            return "series_middle"

        # Check tomorrow's matchup
        tomorrow = self._get_tomorrow_date(game_date)

        try:
            tomorrow_games = self.fetcher.get_schedule(date=tomorrow)

            # Check if same teams play tomorrow
            for next_game in tomorrow_games:
                next_home = next_game.get("home_team")
                next_away = next_game.get("away_team")

                # Same matchup (either direction)
                if ({next_home, next_away} == {home_team, away_team}):
                    logger.info(f"Series continues: {away_team} @ {home_team}")
                    return "series_middle"

            # No matching game tomorrow = series end
            logger.info(f"Series ends: {away_team} @ {home_team}")
            return "series_end"

        except Exception as e:
            logger.error(f"Error checking tomorrow's schedule: {e}")
            return "series_middle"  # Safe default

    def get_next_opponents(
        self,
        team1: str,
        team2: str,
        after_date: str
    ) -> dict:
        """
        Find next opponents for both teams after series ends.

        Used for series_end videos to preview upcoming matchups.

        Args:
            team1: First team name/abbreviation
            team2: Second team name/abbreviation
            after_date: Date string (YYYY-MM-DD) to search from

        Returns:
            {
                "team1_name": {
                    "opponent": "Padres",
                    "game_date": "2024-07-09",
                    "game_time": "19:10",
                    "venue": "Petco Park",
                    "is_home": False,
                    "probable_pitcher": {...}
                },
                "team2_name": {...}
            }
        """
        result = {}

        for team in [team1, team2]:
            next_game = self._find_next_game(team, after_date)
            if next_game:
                result[team] = {
                    "opponent": self._get_opponent(next_game, team),
                    "game_date": next_game.get("date"),
                    "game_time": next_game.get("time"),
                    "venue": next_game.get("venue"),
                    "is_home": next_game.get("home_team") == team,
                    "probable_pitcher": next_game.get("probable_pitchers", {}),
                }
            else:
                result[team] = None

        return result

    def get_series_summary(self, game: dict) -> dict:
        """
        Get summary statistics for the completed series.

        Called when video_type is "series_end".

        Args:
            game: Final game of the series

        Returns:
            {
                "series_score": "2-1",
                "series_winner": "Dodgers",
                "games": [
                    {"date": "2024-07-06", "score": "5-3", "winner": "Dodgers"},
                    {"date": "2024-07-07", "score": "4-2", "winner": "Yankees"},
                    {"date": "2024-07-08", "score": "6-1", "winner": "Dodgers"}
                ],
                "series_mvp": {
                    "player": "Mookie Betts",
                    "stats": "8-13, 2 HR, 5 RBI"
                },
                "key_stats": {
                    "total_runs": {"home": 12, "away": 9},
                    "total_hits": {"home": 28, "away": 22},
                    ...
                }
            }
        """
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        game_date = game.get("date")

        # Find all games in this series
        series_games = self._get_series_games(home_team, away_team, game_date)

        if not series_games:
            return {
                "series_score": "Unknown",
                "series_winner": None,
                "games": [],
                "series_mvp": None,
                "key_stats": {},
            }

        # Calculate series score
        home_wins = sum(1 for g in series_games if g.get("home_score", 0) > g.get("away_score", 0))
        away_wins = len(series_games) - home_wins

        series_winner = home_team if home_wins > away_wins else away_team
        series_score = f"{max(home_wins, away_wins)}-{min(home_wins, away_wins)}"

        return {
            "series_score": series_score,
            "series_winner": series_winner,
            "games": [
                {
                    "date": g.get("date"),
                    "score": f"{g.get('home_score')}-{g.get('away_score')}",
                    "winner": home_team if g.get("home_score", 0) > g.get("away_score", 0) else away_team
                }
                for g in series_games
            ],
            "series_mvp": self._find_series_mvp(series_games),
            "key_stats": self._calculate_series_stats(series_games),
        }

    def get_series_position(self, game: dict) -> dict:
        """
        Determine position within the current series.

        Args:
            game: Current game data

        Returns:
            {
                "game_number": 2,  # 1st, 2nd, or 3rd game
                "series_length": 3,
                "current_score": "1-0",  # Series score before this game
                "is_deciding_game": False
            }
        """
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        game_date = game.get("date")

        # Get previous games in series
        prev_games = self._get_previous_series_games(home_team, away_team, game_date)

        game_number = len(prev_games) + 1

        # Count wins before this game
        home_wins = sum(1 for g in prev_games if g.get("home_score", 0) > g.get("away_score", 0))
        away_wins = len(prev_games) - home_wins

        # Check if it's a potential deciding game
        is_deciding = (home_wins == 1 and away_wins == 1 and game_number == 3)

        return {
            "game_number": game_number,
            "series_length": self._estimate_series_length(home_team, away_team, game_date),
            "current_score": f"{max(home_wins, away_wins)}-{min(home_wins, away_wins)}",
            "is_deciding_game": is_deciding,
        }

    # Helper methods
    def _get_tomorrow_date(self, date_str: str) -> str:
        """Get tomorrow's date string."""
        date = datetime.strptime(date_str, "%Y-%m-%d")
        tomorrow = date + timedelta(days=1)
        return tomorrow.strftime("%Y-%m-%d")

    def _find_next_game(self, team: str, after_date: str) -> Optional[dict]:
        """Find next scheduled game for a team."""
        try:
            # Look ahead up to 7 days
            search_date = datetime.strptime(after_date, "%Y-%m-%d")

            for i in range(1, 8):
                next_date = (search_date + timedelta(days=i)).strftime("%Y-%m-%d")
                games = self.fetcher.get_schedule(date=next_date)

                for game in games:
                    if team in [game.get("home_team"), game.get("away_team")]:
                        return game

            return None
        except Exception as e:
            logger.error(f"Error finding next game for {team}: {e}")
            return None

    def _get_opponent(self, game: dict, team: str) -> str:
        """Get opponent team from game data."""
        if game.get("home_team") == team:
            return game.get("away_team")
        return game.get("home_team")

    def _get_series_games(
        self,
        home_team: str,
        away_team: str,
        end_date: str
    ) -> list[dict]:
        """Get all games in the current series."""
        games = []

        try:
            # Look back up to 5 days for series games
            end = datetime.strptime(end_date, "%Y-%m-%d")

            for i in range(5):
                check_date = (end - timedelta(days=i)).strftime("%Y-%m-%d")
                daily_games = self.fetcher.get_schedule(date=check_date)

                for game in daily_games:
                    # Check if same matchup
                    if {game.get("home_team"), game.get("away_team")} == {home_team, away_team}:
                        games.append(game)

            # Sort by date
            games.sort(key=lambda g: g.get("date", ""))

            return games
        except Exception as e:
            logger.error(f"Error getting series games: {e}")
            return []

    def _get_previous_series_games(
        self,
        home_team: str,
        away_team: str,
        current_date: str
    ) -> list[dict]:
        """Get previous games in this series (before current game)."""
        all_games = self._get_series_games(home_team, away_team, current_date)
        return [g for g in all_games if g.get("date") < current_date]

    def _estimate_series_length(
        self,
        home_team: str,
        away_team: str,
        game_date: str
    ) -> int:
        """Estimate total series length (usually 3, sometimes 2 or 4)."""
        # Default to 3 for most series
        return 3

    def _find_series_mvp(self, series_games: list[dict]) -> Optional[dict]:
        """Find the most valuable player of the series."""
        # TODO: Implement with actual stats aggregation
        return None

    def _calculate_series_stats(self, series_games: list[dict]) -> dict:
        """Calculate aggregate stats for the series."""
        total_home_runs = 0
        total_away_runs = 0

        for game in series_games:
            total_home_runs += game.get("home_score", 0)
            total_away_runs += game.get("away_score", 0)

        return {
            "total_runs": {
                "home": total_home_runs,
                "away": total_away_runs,
            },
        }
