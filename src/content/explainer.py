"""
MLB Video Pipeline - Prediction Explainer

Generates human-readable explanations for AI predictions.
Finds and ranks the top reasons why a prediction was made.

Usage:
    from src.content.explainer import PredictionExplainer

    explainer = PredictionExplainer()
    reasons = explainer.generate_reasoning(player, prediction, context)
"""

from typing import Optional
from dataclasses import dataclass

from src.utils.logger import get_logger
from src.data.predictor_data import PredictionDataProcessor
from config.league_config import PREDICTION_CONFIG


logger = get_logger(__name__)


@dataclass
class PredictionReason:
    """A single reason for a prediction."""
    reason: str              # Human-readable explanation
    strength: float          # How strong this factor is (0-1)
    category: str            # Category of reason
    stat_value: Optional[str] = None  # The actual stat value
    comparison: Optional[str] = None  # Comparison context (league avg, etc.)


class PredictionExplainer:
    """
    Generate ranked reasoning for AI predictions.

    Analyzes multiple factors and returns the top 3 most
    compelling reasons for a prediction.

    Factor Categories:
    - recent_form: Last 10 games performance (weight: 0.30)
    - matchup_history: vs this pitcher/team (weight: 0.25)
    - home_advantage: Home/road splits (weight: 0.15)
    - platoon: L/R advantage (weight: 0.15)
    - pitch_type: Pitch type matchups (weight: 0.10)
    - milestone: Approaching milestone (weight: 0.05)
    """

    def __init__(self, data_processor: Optional[PredictionDataProcessor] = None):
        """
        Initialize the explainer.

        Args:
            data_processor: PredictionDataProcessor for fetching stats
        """
        self.data_processor = data_processor or PredictionDataProcessor()
        self.reasoning_weights = PREDICTION_CONFIG.get("reasoning_types", {})

    def generate_reasoning(
        self,
        player: str,
        prediction: dict,
        game_context: dict
    ) -> list[dict]:
        """
        Find and rank prediction reasons.

        Args:
            player: Player name or ID
            prediction: Model prediction output
                {
                    "class": "above_average" | "average" | "below_average",
                    "confidence": 0.78,
                    "probabilities": [0.12, 0.10, 0.78]
                }
            game_context: Context about the game
                {
                    "opponent_pitcher": {...},
                    "is_home": True,
                    "game_date": "2024-07-08",
                    ...
                }

        Returns:
            Top 3 reasons sorted by strength:
            [
                {
                    "reason": "최근 10경기 타율 .380",
                    "strength": 0.85,
                    "category": "recent_form",
                    "stat_value": ".380",
                    "comparison": "시즌 평균 .290 대비 +90포인트"
                },
                ...
            ]
        """
        logger.info(f"Generating prediction reasoning for {player}")

        all_reasons = []

        # Analyze each factor category
        all_reasons.extend(self._analyze_recent_form(player, prediction, game_context))
        all_reasons.extend(self._analyze_matchup_history(player, prediction, game_context))
        all_reasons.extend(self._analyze_home_away(player, prediction, game_context))
        all_reasons.extend(self._analyze_platoon(player, prediction, game_context))
        all_reasons.extend(self._analyze_pitch_types(player, prediction, game_context))
        all_reasons.extend(self._analyze_milestones(player, prediction, game_context))

        # Sort by strength and return top 3
        sorted_reasons = sorted(all_reasons, key=lambda x: x.strength, reverse=True)
        top_reasons = sorted_reasons[:3]

        # Convert to dict format
        return [
            {
                "reason": r.reason,
                "strength": r.strength,
                "category": r.category,
                "stat_value": r.stat_value,
                "comparison": r.comparison,
            }
            for r in top_reasons
        ]

    def _analyze_recent_form(
        self,
        player: str,
        prediction: dict,
        context: dict
    ) -> list[PredictionReason]:
        """
        Analyze player's recent form (last 10 games).

        Looks for:
        - Hot streak (batting .350+ in last 10)
        - Cold streak (batting .200 or below)
        - Power surge (multiple HRs recently)
        - Slump breaking out
        """
        reasons = []
        base_weight = self.reasoning_weights.get("recent_form", {}).get("weight", 0.30)

        # TODO: Get actual recent stats
        recent_avg = 0.380  # Placeholder
        season_avg = 0.290  # Placeholder

        if recent_avg >= 0.350:
            strength = min(1.0, base_weight + (recent_avg - 0.350) * 2)
            reasons.append(PredictionReason(
                reason=f"최근 10경기 타율 {recent_avg:.3f}",
                strength=strength,
                category="recent_form",
                stat_value=f".{int(recent_avg*1000)}",
                comparison=f"시즌 평균 {season_avg:.3f} 대비 +{int((recent_avg-season_avg)*1000)}포인트"
            ))

        elif recent_avg <= 0.200:
            strength = min(1.0, base_weight + (0.200 - recent_avg) * 2)
            reasons.append(PredictionReason(
                reason=f"최근 10경기 타율 {recent_avg:.3f}로 부진",
                strength=strength,
                category="recent_form",
                stat_value=f".{int(recent_avg*1000)}",
                comparison=f"시즌 평균 {season_avg:.3f} 대비 -{int((season_avg-recent_avg)*1000)}포인트"
            ))

        return reasons

    def _analyze_matchup_history(
        self,
        player: str,
        prediction: dict,
        context: dict
    ) -> list[PredictionReason]:
        """
        Analyze batter vs pitcher history.

        Looks for:
        - Strong history (10+ ABs, .300+ avg)
        - Weak history (10+ ABs, .200 or below)
        - First time facing pitcher
        """
        reasons = []
        base_weight = self.reasoning_weights.get("matchup_history", {}).get("weight", 0.25)

        pitcher = context.get("opponent_pitcher", {})
        if not pitcher:
            return reasons

        # TODO: Get actual matchup data
        matchup_stats = self.data_processor.get_matchup_features(
            player,
            pitcher.get("id", ""),
            context.get("game_date", "")
        )

        abs_count = matchup_stats.get("at_bats", 0)
        avg = matchup_stats.get("batting_average", 0.0)

        if abs_count >= 10:
            if avg >= 0.300:
                strength = min(1.0, base_weight + (avg - 0.300) * 1.5)
                reasons.append(PredictionReason(
                    reason=f"{pitcher.get('name', '상대 투수')} 상대 통산 {avg:.3f}",
                    strength=strength,
                    category="matchup_history",
                    stat_value=f"{abs_count}타수 {int(abs_count * avg)}안타",
                    comparison=f"리그 평균 대비 우수한 상대 전적"
                ))
            elif avg <= 0.200:
                strength = min(1.0, base_weight + (0.200 - avg) * 1.5)
                reasons.append(PredictionReason(
                    reason=f"{pitcher.get('name', '상대 투수')} 상대 통산 {avg:.3f}로 고전",
                    strength=strength,
                    category="matchup_history",
                    stat_value=f"{abs_count}타수 {int(abs_count * avg)}안타",
                    comparison=f"상대 전적 부진"
                ))

        return reasons

    def _analyze_home_away(
        self,
        player: str,
        prediction: dict,
        context: dict
    ) -> list[PredictionReason]:
        """
        Analyze home/away splits.

        Significant if difference is >= 50 points in batting average.
        """
        reasons = []
        base_weight = self.reasoning_weights.get("home_advantage", {}).get("weight", 0.15)

        is_home = context.get("is_home", False)

        # TODO: Get actual split data
        splits = self.data_processor.get_split_features(
            player,
            "home_away",
            context.get("season", 2024)
        )

        home_avg = splits.get("primary", {}).get("avg", 0.0)
        away_avg = splits.get("secondary", {}).get("avg", 0.0)
        diff = abs(home_avg - away_avg)

        if diff >= 0.050:  # Significant split
            if is_home and home_avg > away_avg:
                strength = min(1.0, base_weight + diff)
                reasons.append(PredictionReason(
                    reason=f"홈에서 더 강한 타자 (홈 {home_avg:.3f} vs 원정 {away_avg:.3f})",
                    strength=strength,
                    category="home_advantage",
                    stat_value=f"+{int(diff*1000)} 포인트",
                    comparison="홈 경기 이점"
                ))
            elif not is_home and away_avg > home_avg:
                strength = min(1.0, base_weight + diff)
                reasons.append(PredictionReason(
                    reason=f"원정에서 오히려 강한 타자 (원정 {away_avg:.3f} vs 홈 {home_avg:.3f})",
                    strength=strength,
                    category="home_advantage",
                    stat_value=f"+{int(diff*1000)} 포인트",
                    comparison="원정 경기에서 호조"
                ))

        return reasons

    def _analyze_platoon(
        self,
        player: str,
        prediction: dict,
        context: dict
    ) -> list[PredictionReason]:
        """
        Analyze platoon advantage (L/R splits).

        Looks for significant advantage when batter has
        opposite handedness from pitcher.
        """
        reasons = []
        base_weight = self.reasoning_weights.get("platoon", {}).get("weight", 0.15)

        pitcher = context.get("opponent_pitcher", {})
        pitcher_hand = pitcher.get("throws", "R")

        # TODO: Get player handedness and splits
        # For now, placeholder logic
        vs_left = 0.320
        vs_right = 0.280

        if pitcher_hand == "L" and vs_left > vs_right + 0.030:
            strength = min(1.0, base_weight + (vs_left - vs_right))
            reasons.append(PredictionReason(
                reason=f"좌완 투수 상대로 강한 타격 ({vs_left:.3f})",
                strength=strength,
                category="platoon",
                stat_value=f"좌완 상대 {vs_left:.3f}",
                comparison=f"우완 상대 {vs_right:.3f} 대비 우수"
            ))

        return reasons

    def _analyze_pitch_types(
        self,
        player: str,
        prediction: dict,
        context: dict
    ) -> list[PredictionReason]:
        """
        Analyze player's performance vs specific pitch types.

        Useful when pitcher has a dominant pitch that the
        batter handles well (or poorly).
        """
        reasons = []
        # Lower priority, implement with Statcast data
        return reasons

    def _analyze_milestones(
        self,
        player: str,
        prediction: dict,
        context: dict
    ) -> list[PredictionReason]:
        """
        Check for approaching milestones.

        Examples:
        - 2 hits away from 100th career hit
        - Needs 1 HR for 30-HR season
        - Approaching .300 average milestone
        """
        reasons = []
        base_weight = self.reasoning_weights.get("milestone", {}).get("weight", 0.05)

        # TODO: Get actual milestone data
        # Placeholder: would check current stats vs round numbers

        return reasons

    def format_reasoning_for_script(self, reasons: list[dict]) -> str:
        """
        Format reasons for video script narration.

        Args:
            reasons: List of reason dictionaries

        Returns:
            Formatted string for narration
        """
        if not reasons:
            return "다양한 요소들을 종합적으로 분석한 결과입니다."

        formatted = []
        ordinals = ["첫째,", "둘째,", "셋째,"]

        for i, reason in enumerate(reasons[:3]):
            prefix = ordinals[i] if i < len(ordinals) else f"{i+1}."
            formatted.append(f"{prefix} {reason['reason']}")

        return " ".join(formatted)

    def get_confidence_explanation(self, confidence: float) -> str:
        """
        Explain the confidence level in natural language.

        Args:
            confidence: Model confidence (0-1)

        Returns:
            Korean explanation of confidence level
        """
        if confidence >= 0.85:
            return "AI가 매우 높은 확신을 가지고 예측합니다"
        elif confidence >= 0.70:
            return "AI가 높은 확률로 예측합니다"
        elif confidence >= 0.55:
            return "AI가 조심스럽게 예측합니다"
        else:
            return "경기 결과가 쉽게 예측되지 않습니다"
