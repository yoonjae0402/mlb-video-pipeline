"""
CLI tool to fetch MLB game data using the MLBDataFetcher.

Usage:
    python scripts/fetch_games.py --date 2024-07-04
    python scripts/fetch_games.py --date 2024-07-04 --output custom_name.json
    python scripts/fetch_games.py --date 2024-07-04 --verbose
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from datetime import datetime

from src.data.fetcher import MLBDataFetcher
from src.utils.logger import get_logger
from src.utils.exceptions import DataFetchError

# Configure logger for this script
logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Fetch MLB game data for a given date."
    )
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Date to fetch games for in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional: Output filename (e.g., 'yankees_redsox_20240704.json'). "
             "Defaults to 'mlb_games_YYYYMMDD.json'."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase output verbosity (shows fetched data on console)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Directory to save the output JSON file. Defaults to 'data/raw'."
    )

    args = parser.parse_args()

    fetcher = MLBDataFetcher()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    try:
        games_data = fetcher.fetch_games(args.date)

        if not games_data:
            logger.info(f"No games found for {args.date}.")
            return

        if args.output:
            filename = args.output
            if not filename.lower().endswith(".json"):
                filename += ".json"
        else:
            filename = f"mlb_games_{args.date.replace('-', '')}.json"
            
        full_output_path = output_path / filename

        with open(full_output_path, "w", encoding="utf-8") as f:
            json.dump(games_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully fetched {len(games_data)} games for {args.date} "
                    f"and saved to {full_output_path}")

        if args.verbose:
            print("\n--- Fetched Game Data (Verbose Output) ---")
            print(json.dumps(games_data, indent=2, ensure_ascii=False))
            print("------------------------------------------")

    except DataFetchError as e:
        logger.error(f"Error fetching MLB game data: {e}", exc_info=True)
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
