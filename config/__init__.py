"""
MLB Video Pipeline - Configuration Package

This package contains all configuration settings for the pipeline:
- settings.py: Core application settings (paths, API keys, limits)
- league_config.py: MLB-specific configurations (teams, positions, stats)

Usage:
    from config.settings import Settings
    settings = Settings()

    from config.league_config import MLB_TEAMS, STAT_CATEGORIES
"""

from config.settings import Settings

__all__ = ["Settings"]
