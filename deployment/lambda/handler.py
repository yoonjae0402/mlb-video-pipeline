"""
MLB Video Pipeline - AWS Lambda Handler

Serverless function for running pipeline components on AWS Lambda.

Deploy with AWS SAM or Serverless Framework.
"""

import json
import os
import sys
from datetime import datetime, timedelta

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def fetch_games_handler(event, context):
    """
    Lambda handler for fetching MLB games.

    Event format:
    {
        "date": "2024-07-04",  // Optional, defaults to yesterday
        "team": "NYY"          // Optional team filter
    }
    """
    from src.data.fetcher import MLBDataFetcher
    from src.data.processor import DataProcessor

    # Parse event
    date = event.get("date")
    if not date:
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    team = event.get("team")

    try:
        fetcher = MLBDataFetcher()
        games = fetcher.get_games_for_date(date)

        # Filter by team if specified
        if team:
            from config.league_config import get_team_by_abbreviation
            result = get_team_by_abbreviation(team)
            if result:
                team_id, _ = result
                games = [
                    g for g in games
                    if g.get("home_team_id") == team_id or g.get("away_team_id") == team_id
                ]

        # Process and save
        processor = DataProcessor()
        df = processor.process_games(games)
        filepath = processor.save_processed_data(df, f"games_{date}")

        return {
            "statusCode": 200,
            "body": json.dumps({
                "date": date,
                "games_found": len(games),
                "file": str(filepath),
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }


def generate_script_handler(event, context):
    """
    Lambda handler for generating video scripts.

    Event format:
    {
        "game_data": {...},
        "duration": 60
    }
    """
    from src.content.script_generator import ScriptGenerator

    game_data = event.get("game_data", {})
    duration = event.get("duration", 60)

    try:
        generator = ScriptGenerator()
        script = generator.generate_game_recap(game_data, duration=duration)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "script": script,
                "length": len(script),
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }


def health_check_handler(event, context):
    """
    Lambda handler for health checks.
    """
    from config.settings import settings

    api_keys = settings.validate_api_keys()

    return {
        "statusCode": 200,
        "body": json.dumps({
            "status": "healthy",
            "api_keys": api_keys,
            "timestamp": datetime.now().isoformat(),
        })
    }
