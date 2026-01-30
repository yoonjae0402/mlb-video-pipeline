"""
MLB Video Pipeline - Post-game Analysis Module

Analyzes completed games for video content generation.
Extracts storylines, key performances, and contextual insights
WITHOUT making predictions.

Usage:
    from src.data.analyzer import MLBStatsAnalyzer

    analyzer = MLBStatsAnalyzer()
    insights = analyzer.analyze_game(game_data)
"""

from typing import Any, Optional
from datetime import datetime

from src.utils.logger import get_logger
from src.data.fetcher import MLBDataFetcher


logger = get_logger(__name__)


class MLBStatsAnalyzer:
    """
    Generate insights from completed MLB games.

    Responsibilities:
    - Extract key storylines (streaks, milestones, comebacks)
    - Find interesting statistics and anomalies
    - Analyze player performance highlights
    - Provide season context and standings impact

    Note: This class does NOT make predictions. It only analyzes
    completed games for content generation.
    """

    def __init__(self, fetcher: Optional[MLBDataFetcher] = None):
        """
        Initialize the analyzer.

        Args:
            fetcher: Optional MLBDataFetcher instance for additional data
        """
        self.fetcher = fetcher or MLBDataFetcher()

    def analyze_game(self, game_data: dict) -> dict:
        """
        Analyze a completed game for content generation.

        Args:
            game_data: Game data dictionary from fetcher

        Returns:
            {
                "key_insights": [str, ...],  # Main talking points
                "top_performances": [dict, ...],  # Standout players
                "storylines": [str, ...],  # Narrative angles
                "season_context": {
                    "standings_impact": str,
                    "playoff_implications": str,
                    "streak_info": str
                },
                "game_flow": {
                    "turning_point": str,
                    "decisive_moment": str,
                    "momentum_shifts": int
                }
            }
        """
        logger.info(f"Analyzing game: {game_data.get('game_id')}")

        return {
            "key_insights": self._extract_key_insights(game_data),
            "top_performances": self._find_top_performances(game_data),
            "storylines": self.find_storylines(game_data),
            "season_context": self._get_season_context(game_data),
            "game_flow": self._analyze_game_flow(game_data),
        }

    def find_storylines(self, game_data: dict) -> list[str]:
        """
        Find interesting narratives from the game.

        Looks for:
        - Hitting/winning/losing streaks
        - Milestone achievements (100 RBIs, 20 wins, etc.)
        - Comeback victories
        - Pitcher duels / blowouts
        - Rookie performances
        - Player vs former team
        - Historical comparisons

        Args:
            game_data: Game data dictionary

        Returns:
            List of storyline descriptions
        """
        storylines = []

        # Check for comeback
        if self._is_comeback(game_data):
            storylines.append(self._describe_comeback(game_data))

        # Check for milestone performances
        milestones = self._find_milestones(game_data)
        storylines.extend(milestones)

        # Check for streaks
        streaks = self._find_streaks(game_data)
        storylines.extend(streaks)

        # Check for notable matchups
        matchups = self._find_notable_matchups(game_data)
        storylines.extend(matchups)

        return storylines[:5]  # Limit to top 5 storylines

    def _extract_key_insights(self, game_data: dict) -> list[str]:
        """Extract the most important talking points from the game."""
        insights = []

        # Score differential analysis
        home_score = game_data.get("home_score", 0)
        away_score = game_data.get("away_score", 0)
        diff = abs(home_score - away_score)

        if diff >= 7:
            winner = game_data.get("home_team" if home_score > away_score else "away_team")
            insights.append(f"Dominant victory by {winner} with a {diff}-run margin")
        elif diff <= 1:
            insights.append("Nail-biting finish decided by a single run")

        # High-scoring game
        total_runs = home_score + away_score
        if total_runs >= 15:
            insights.append(f"Offensive explosion with {total_runs} combined runs")

        # Pitching duel
        if total_runs <= 4 and home_score <= 2 and away_score <= 2:
            insights.append("Classic pitching duel with dominant performances on the mound")

        return insights

    def _find_top_performances(self, game_data: dict) -> list[dict]:
        """
        Find standout individual performances.

        Returns list of player performance dictionaries.
        """
        performances = []

        # Analyze box score if available
        box_score = game_data.get("box_score", {})

        # Find batting stars
        for team in ["home", "away"]:
            batters = box_score.get(f"{team}_batters", [])
            for batter in batters:
                # Multi-hit game
                if batter.get("hits", 0) >= 3:
                    performances.append({
                        "player": batter.get("name"),
                        "team": game_data.get(f"{team}_team"),
                        "type": "batting",
                        "highlight": f"{batter.get('hits')} hits",
                        "stats": batter,
                    })
                # Multi-homer game
                if batter.get("home_runs", 0) >= 2:
                    performances.append({
                        "player": batter.get("name"),
                        "team": game_data.get(f"{team}_team"),
                        "type": "power",
                        "highlight": f"{batter.get('home_runs')} home runs",
                        "stats": batter,
                    })

        # Sort by impact and return top performers
        return sorted(performances,
                     key=lambda x: self._calculate_impact_score(x),
                     reverse=True)[:3]

    def _get_season_context(self, game_data: dict) -> dict:
        """Get broader season context for the game."""
        return {
            "standings_impact": self._calculate_standings_impact(game_data),
            "playoff_implications": self._check_playoff_implications(game_data),
            "streak_info": self._get_streak_info(game_data),
        }

    def _analyze_game_flow(self, game_data: dict) -> dict:
        """Analyze how the game unfolded."""
        return {
            "turning_point": self._find_turning_point(game_data),
            "decisive_moment": self._find_decisive_moment(game_data),
            "momentum_shifts": self._count_momentum_shifts(game_data),
        }

    # Helper methods
    def _is_comeback(self, game_data: dict) -> bool:
        """Check if the game featured a comeback."""
        scoring = game_data.get("scoring_plays", [])
        if not scoring:
            return False

        max_deficit = 0
        current_diff = 0

        for play in scoring:
            runs = play.get("runs", 1)
            is_home = play.get("is_home_team", False)
            current_diff += runs if is_home else -runs
            max_deficit = max(max_deficit, abs(current_diff))

        final_winner_is_home = game_data.get("home_score", 0) > game_data.get("away_score", 0)
        return max_deficit >= 3

    def _describe_comeback(self, game_data: dict) -> str:
        """Describe a comeback victory."""
        winner = game_data.get("home_team" if
                               game_data.get("home_score", 0) > game_data.get("away_score", 0)
                               else "away_team")
        return f"{winner} stages an incredible comeback victory"

    def _find_milestones(self, game_data: dict) -> list[str]:
        """Find milestone achievements in the game."""
        milestones = []
        # Implementation would check for round-number achievements
        # e.g., 100th RBI, 200th hit, 50th save, etc.
        return milestones

    def _find_streaks(self, game_data: dict) -> list[str]:
        """Find notable streaks extended or broken."""
        streaks = []
        # Implementation would track winning/hitting streaks
        return streaks

    def _find_notable_matchups(self, game_data: dict) -> list[str]:
        """Find notable individual matchups."""
        matchups = []
        # e.g., player vs former team, rivalry matchups
        return matchups

    def _calculate_impact_score(self, performance: dict) -> float:
        """Calculate impact score for ranking performances."""
        score = 0.0
        stats = performance.get("stats", {})

        score += stats.get("hits", 0) * 1.0
        score += stats.get("home_runs", 0) * 3.0
        score += stats.get("rbi", 0) * 1.5
        score += stats.get("runs", 0) * 1.0

        return score

    def _calculate_standings_impact(self, game_data: dict) -> str:
        """Calculate impact on division/wild card standings."""
        return "To be implemented with standings data"

    def _check_playoff_implications(self, game_data: dict) -> str:
        """Check playoff race implications."""
        return "To be implemented with standings data"

    def _get_streak_info(self, game_data: dict) -> str:
        """Get current streak information for both teams."""
        return ""

    def _find_turning_point(self, game_data: dict) -> str:
        """Find the game's turning point."""
        return "To be implemented with play-by-play data"

    def _find_decisive_moment(self, game_data: dict) -> str:
        """Find the game's decisive moment."""
        return "To be implemented with play-by-play data"

    def _count_momentum_shifts(self, game_data: dict) -> int:
        """Count significant momentum shifts."""
        return 0
