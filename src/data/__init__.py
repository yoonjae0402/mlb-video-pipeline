"""
MLB Video Pipeline - Data Package

Handles all MLB data operations:
- fetcher: Collect data from MLB Stats API
- analyzer: Post-game analysis for content generation
- predictor_data: Prepare data for prediction models
- series_tracker: Track series progress and video types

Usage:
    from src.data import MLBDataFetcher, MLBStatsAnalyzer, SeriesTracker

    fetcher = MLBDataFetcher()
    games = fetcher.get_games_for_date("2024-07-04")

    analyzer = MLBStatsAnalyzer()
    insights = analyzer.analyze_game(games[0])

    tracker = SeriesTracker()
    video_type = tracker.get_video_type(games[0])
"""

from src.data.fetcher import MLBDataFetcher
from src.data.analyzer import MLBStatsAnalyzer
from src.data.predictor_data import PredictionDataProcessor
from src.data.series_tracker import SeriesTracker

__all__ = [
    "MLBDataFetcher",
    "MLBStatsAnalyzer",
    "PredictionDataProcessor",
    "SeriesTracker",
]
