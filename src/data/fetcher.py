import json
import os
import time
import datetime
from datetime import timedelta
import logging
from typing import List, Dict, Any, Optional, Union

import statsapi
import requests

from src.utils.logger import get_logger
from src.utils.exceptions import DataFetchError # Assuming a custom exception for data fetching

logger = get_logger(__name__)

class MLBDataFetcher:
    """
    Fetches and processes MLB game data from the MLB-StatsAPI.
    
    Responsibilities:
    - Fetch games for specific dates
    - Extract relevant information (scores, plays, performers)
    - Handle API errors and retries
    - Cache results to avoid duplicate calls
    - Validate data before returning
    """
    
    def __init__(self, cache_dir: str = "data/cache", timeout: int = 30):
        """
        Initialize with optional caching and request timeout.

        Args:
            cache_dir (str): Directory to store cached game data.
            timeout (int): Timeout for API requests in seconds.
        """
        self.cache_dir = cache_dir
        self.timeout = timeout
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"MLBDataFetcher initialized with cache_dir: {self.cache_dir}")

    def _get_cache_file_path(self, date: str, game_id: Optional[int] = None) -> str:
        """Helper to construct cache file path."""
        if game_id:
            return os.path.join(self.cache_dir, f"game_detail_{game_id}.json")
        return os.path.join(self.cache_dir, f"schedule_{date}.json")

    def _load_from_cache(self, date: str, game_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Loads data from cache if available and valid.
        
        Args:
            date (str): Date string (YYYY-MM-DD).
            game_id (Optional[int]): Game ID for detailed game cache.

        Returns:
            Optional[Dict[str, Any]]: Cached data or None if not found/invalid.
        """
        file_path = self._get_cache_file_path(date, game_id)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Defensive check: Ensure loaded data is a dictionary
                if not isinstance(data, dict):
                    logger.warning(f"Cached file {file_path} contains non-dictionary data (type: {type(data)}). Deleting and treating as cache miss.")
                    os.remove(file_path)
                    return None

                if not game_id and not self._should_use_cache(date, data):
                    logger.info(f"Cache for schedule {date} is stale. Deleting: {file_path}")
                    os.remove(file_path)
                    return None
                
                logger.info(f"Cache hit for {'game_id' if game_id else 'schedule'} {game_id if game_id else date}")
                return data
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding cached file {file_path}: {e}")
                os.remove(file_path) # Corrupted cache, delete it
            except Exception as e:
                logger.error(f"Unexpected error loading cache {file_path}: {e}")
        logger.debug(f"Cache miss for {'game_id' if game_id else 'schedule'} {game_id if game_id else date}")
        return None

    def _save_to_cache(self, data: Dict[str, Any], date: str, game_id: Optional[int] = None) -> None:
        """Saves data to cache."""
        file_path = self._get_cache_file_path(date, game_id)
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            logger.debug(f"Saved to cache: {file_path}")
        except Exception as e:
            logger.error(f"Error saving to cache {file_path}: {e}")

    def _should_use_cache(self, date_str: str, cached_schedule: Dict[str, Any]) -> bool:
        """
        Check if cached data for a schedule is still valid based on game status.

        Args:
            date_str (str): Date string (YYYY-MM-DD).
            cached_schedule (Dict[str, Any]): The cached schedule data.

        Returns:
            bool: True if cache is valid, False otherwise.
        """
        if not isinstance(cached_schedule, dict):
            logger.critical(f"CRITICAL ERROR: _should_use_cache received non-dict type for cached_schedule: {type(cached_schedule)}. Value: {cached_schedule}")
            raise TypeError(f"_should_use_cache expected dict, got {type(cached_schedule)}")

        if not cached_schedule or "games" not in cached_schedule:
            return False

        # Schedule cache is valid if all games are final
        all_final = all(game.get("status", {}).get("statusCode") == "F" for game in cached_schedule["games"])
        if all_final:
            return True # Completed games cached permanently

        # For in-progress or scheduled games, check timestamp
        cache_timestamp_str = cached_schedule.get("cache_timestamp")
        if not cache_timestamp_str:
            return False

        cache_timestamp = datetime.datetime.fromisoformat(cache_timestamp_str)
        now = datetime.datetime.now()

        # In-progress games expire after 5 minutes
        if any(game.get("status", {}).get("statusCode") in ["I", "P"] for game in cached_schedule["games"]):
            return (now - cache_timestamp) < timedelta(minutes=5)
        
        # Scheduled games expire after 1 hour
        if any(game.get("status", {}).get("statusCode") in ["S"] for game in cached_schedule["games"]):
            return (now - cache_timestamp) < timedelta(hours=1)
        
        return False # Default to false for unknown statuses or other edge cases

    def _make_api_call(self, api_func, *args, **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Handles API calls with retries and exponential backoff.
        
        Args:
            api_func (callable): The statsapi function to call.
            *args, **kwargs: Arguments to pass to the api_func.

        Returns:
            Union[Dict[str, Any], List[Dict[str, Any]]]: The JSON response from the API (dict or list).

        Raises:
            DataFetchError: If API call fails after retries or returns an unexpected type.
        """
        retries = 3
        backoff_factor = 0.5
        for i in range(retries):
            try:
                logger.debug(f"Attempt {i+1}/{retries} for API call: {api_func.__name__} with args: {args}, kwargs: {kwargs}")
                response = api_func(*args, **kwargs)

                # Explicitly check response type: should be dict or list
                if not isinstance(response, (dict, list, type(None))): # statsapi can return None for no results
                    logger.error(f"API call to {api_func.__name__} returned unexpected type: {type(response)}. Value: {response}")
                    raise DataFetchError(f"API returned unexpected non-JSON type: {type(response)}", source="MLB-StatsAPI", game_id=kwargs.get('gamePk'))

                if response is not None: # Use 'is not None' instead of just 'if response' as empty dict/list are valid
                    logger.debug(f"API call successful, type of response: {type(response)}")
                    return response
                else:
                    # If response is None, it means no data, not an error that needs retry
                    logger.info(f"API call to {api_func.__name__} returned None (no data).")
                    return {} if api_func == statsapi.get else [] # Return appropriate empty type
            except requests.exceptions.Timeout:
                logger.warning(f"API call timed out. Retrying in {backoff_factor * (2 ** i)} seconds...")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error during API call. Retrying in {backoff_factor * (2 ** i)} seconds...")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request exception during API call: {e}. Retrying in {backoff_factor * (2 ** i)} seconds...")
            except json.JSONDecodeError:
                logger.warning(f"Malformed JSON response from API. Retrying in {backoff_factor * (2 ** i)} seconds...")
            except Exception as e:
                logger.error(f"Unexpected error during API call: {e}. Retrying in {backoff_factor * (2 ** i)} seconds...")
            
            time.sleep(backoff_factor * (2 ** i))
        
        logger.error(f"Failed API call after {retries} retries: {api_func.__name__}")
        raise DataFetchError(f"Failed to fetch data from API after {retries} retries.", source="MLB-StatsAPI", game_id=kwargs.get('gamePk'))

    def fetch_games(self, date_str: str) -> List[Dict[str, Any]]:
        """
        Fetch all games for a given date.
        
        Args:
            date_str (str): Date string in format "YYYY-MM-DD".
            
        Returns:
            List[Dict[str, Any]]: List of game dictionaries with standardized structure.
            
        Raises:
            DataFetchError: If API call fails after retries or date format is invalid.
        """
        try:
            date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            raise DataFetchError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD.", source="MLB-StatsAPI")

        # Try to load from cache
        cached_schedule_wrapper = self._load_from_cache(date_str)
        if cached_schedule_wrapper:
            if self._should_use_cache(date_str, cached_schedule_wrapper):
                cached_schedule_list = cached_schedule_wrapper.get("games", [])
                # If all games are final, return directly.
                if all(game.get("status", {}).get("statusCode") == "F" for game in cached_schedule_list):
                    logger.info(f"Cache hit for schedule {date_str} with all final games. Returning cached data.")
                    return [self._standardize_game_data(game) for game in cached_schedule_list]
                else:
                    logger.info(f"Cached schedule for {date_str} contains non-final games. Re-fetching details for those.")
                    standardized_games_from_cache = []
                    for game_summary in cached_schedule_list:
                        game_id = game_summary['game_id']
                        game_detail = self.get_game_details(game_id) # This will handle its own caching and re-fetching
                        if game_detail:
                            standardized_games_from_cache.append(game_detail)
                    return standardized_games_from_cache
            else:
                logger.info(f"Cached schedule for {date_str} is stale or invalid. Re-fetching full schedule.")
        
        logger.info(f"Fetching games for date: {date_str}")
        try:
            # Use statsapi.schedule for game IDs and basic info, it returns a list
            schedule_data_list = self._make_api_call(statsapi.schedule, date=date_obj)
            
            if not schedule_data_list:
                logger.info(f"No games found for date: {date_str}")
                return []
            
            # Wrap the list in a dictionary to add cache_timestamp for the schedule as a whole
            cacheable_schedule_wrapper = {
                "games": schedule_data_list,
                "cache_timestamp": datetime.datetime.now().isoformat()
            }
            self._save_to_cache(cacheable_schedule_wrapper, date_str) # Cache the wrapped schedule

            standardized_games = []
            for game_summary in schedule_data_list:
                game_id = game_summary['game_id']
                game_detail = self.get_game_details(game_id) # This will handle its own caching
                if game_detail:
                    standardized_games.append(game_detail)
            
            return standardized_games

        except DataFetchError:
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching games for {date_str}: {e}")
            raise DataFetchError(f"Failed to fetch games for {date_str}.", source="MLB-StatsAPI")

    def get_game_details(self, game_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed information for a specific game.
        
        Args:
            game_id (int): The unique identifier for the game.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing standardized game details, or None if fetching fails.
        """
        date_str_for_cache = datetime.datetime.now().strftime("%Y-%m-%d") # Use current date for cache path if game date is unknown
        cached_game_data = self._load_from_cache(date_str_for_cache, game_id) # Check cache using current date as a placeholder
        if cached_game_data and cached_game_data.get("status", {}).get("statusCode") == "F":
            return cached_game_data # If cached and final, return immediately
        elif cached_game_data:
             logger.info(f"Cached game {game_id} is not final, re-fetching details.")


        logger.info(f"Fetching details for game_id: {game_id}")
        try:
            # Using statsapi.get for detailed game data
            game_data = self._make_api_call(statsapi.get, 'game', {'gamePk': game_id})
            
            if not game_data:
                logger.warning(f"No detailed data found for game_id: {game_id}")
                return None

            # Determine the actual game date from the fetched data for accurate caching
            game_date_str = game_data.get("gameData", {}).get("datetime", {}).get("officialDate")
            if not game_date_str:
                game_date_str = datetime.datetime.now().strftime("%Y-%m-%d") # Fallback
                logger.warning(f"Could not determine game date for {game_id}, using current date for cache: {game_date_str}")

            standardized_data = self._standardize_game_data(game_data)
            
            # Add cache timestamp to detailed game data
            standardized_data["cache_timestamp"] = datetime.datetime.now().isoformat()
            self._save_to_cache(standardized_data, game_date_str, game_id)
            
            return standardized_data
        except DataFetchError:
            logger.error(f"Failed to fetch details for game_id {game_id} after retries.")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching details for game_id {game_id}: {e}")
            return None

    def _standardize_game_data(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardizes the raw game data into the defined structure.
        
        Args:
            game_data (Dict[str, Any]): Raw game data from MLB-StatsAPI.

        Returns:
            Dict[str, Any]: Standardized game dictionary.
        """
        # A lot of this will require careful parsing of the statsapi output.
        # This is a placeholder and will need to be thoroughly implemented.

        if not game_data or 'gamePk' not in game_data:
            logger.warning(f"Malformed game_data received for standardization: {game_data}")
            return {}

        game_id = game_data.get('gamePk')
        
        # Determine if it's a schedule summary or full game detail
        is_full_detail = 'liveData' in game_data

        # Extract basic game info
        game_date = game_data.get("gameData", {}).get("datetime", {}).get("officialDate") if is_full_detail else game_data.get("game_date")
        game_status_code = game_data.get("gameData", {}).get("status", {}).get("statusCode") if is_full_detail else game_data.get("status", {}).get("statusCode")
        game_status = game_data.get("gameData", {}).get("status", {}).get("detailedState") if is_full_detail else game_data.get("status", {}).get("detailedState")

        home_team_data = game_data.get("gameData", {}).get("teams", {}).get("home") if is_full_detail else game_data.get("home")
        away_team_data = game_data.get("gameData", {}).get("teams", {}).get("away") if is_full_detail else game_data.get("away")
        
        home_score = game_data.get("liveData", {}).get("linescore", {}).get("teams", {}).get("home", {}).get("runs") if is_full_detail else game_data.get("home_score")
        away_score = game_data.get("liveData", {}).get("linescore", {}).get("teams", {}).get("away", {}).get("runs") if is_full_detail else game_data.get("away_score")

        venue_data = game_data.get("gameData", {}).get("venue") if is_full_detail else game_data.get("venue")
        
        standardized = {
            "game_id": game_id,
            "date": game_date,
            "status": game_status,
            "home_team": {
                "name": home_team_data.get("name") if home_team_data else None,
                "abbreviation": home_team_data.get("abbreviation") if home_team_data else None,
                "score": home_score
            },
            "away_team": {
                "name": away_team_data.get("name") if away_team_data else None,
                "abbreviation": away_team_data.get("abbreviation") if away_team_data else None,
                "score": away_score
            },
            "venue": {
                "name": venue_data.get("name") if venue_data else None,
                # Location often needs a separate lookup or parsing from venue info
                "location": "Unknown" # Placeholder, needs more sophisticated extraction
            },
            "key_plays": [],
            "top_performers": [],
            "game_highlights": {},
            "season_context": {}
        }

        if is_full_detail:
            standardized["key_plays"] = self._extract_key_plays(game_data)
            standardized["top_performers"] = self._extract_top_performers(game_data)
            standardized["game_highlights"] = self._extract_game_highlights(game_data)
            standardized["season_context"] = self._extract_season_context(game_data)
            # Update venue location if possible from full detail
            venue_location = game_data.get("venue", {}).get("location", {}).get("cityState") # Example path
            if venue_location:
                standardized["venue"]["location"] = venue_location

        return standardized

    def _extract_key_plays(self, game_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract and format important plays from detailed game data.
        This is a complex extraction, focusing on high-impact events.
        """
        key_plays = []
        if not game_data or "liveData" not in game_data or "plays" not in game_data["liveData"]:
            return key_plays

        all_plays = game_data["liveData"]["plays"]["allPlays"]
        
        # Simple heuristic for key plays: HRs, important strikeouts, game-changing plays
        for play in all_plays:
            play_events = play.get("playEvents", [])
            for event in play_events:
                event_type = event.get("details", {}).get("event")
                event_description = event.get("description")
                inning = play.get("about", {}).get("inning")

                if not event_type or not event_description or not inning:
                    continue

                impact = "low"
                play_type = "other"

                if "Home Run" in event_type or "homers" in event_description:
                    impact = "high"
                    play_type = "home_run"
                elif "Strikeout" in event_type or "strikes out" in event_description:
                    # Filter for higher impact strikeouts, e.g., bases loaded, late innings
                    if play.get("result", {}).get("rbi") > 0 or inning >= 7: # Placeholder logic
                        impact = "medium"
                        play_type = "strikeout"
                elif "Walk-off" in event_type or "walks off" in event_description:
                    impact = "high"
                    play_type = "walk_off"
                elif play.get("result", {}).get("rbi") >= 2: # 2+ RBI plays
                    impact = "medium"
                    play_type = "rbi_double" if "double" in event_type else "rbi_single"


                if impact != "low": # Only add high/medium impact plays for now
                    key_plays.append({
                        "inning": inning,
                        "description": event_description,
                        "impact": impact,
                        "type": play_type
                    })
        
        return key_plays

    def _extract_top_performers(self, game_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify and format top 3 performers from detailed game data.
        This is a heuristic-based extraction.
        """
        performers = []
        if not game_data or "liveData" not in game_data:
            return performers

        # Focus on players who completed the game and had significant stats
        boxscore = game_data["liveData"].get("boxscore")
        if not boxscore:
            return performers
        
        all_players = boxscore.get("teams", [])
        player_stats = {} # {player_id: {batting_stats, pitching_stats, team_id, position, name}}

        for team_data in all_players:
            team_id = team_data.get("team", {}).get("id")
            team_abbreviation = team_data.get("team", {}).get("abbreviation")
            
            # Batting stats
            players_by_type = team_data.get("players")
            if players_by_type:
                for player_id, player_info in players_by_type.items():
                    if player_info.get("status", {}).get("description") == "Active":
                        full_name = player_info.get("fullName")
                        position = player_info.get("position", {}).get("abbreviation")
                        
                        stats = player_info.get("stats", {})
                        
                        if player_id not in player_stats:
                            player_stats[player_id] = {
                                "name": full_name,
                                "position": position,
                                "team": team_abbreviation,
                                "stats": {},
                                "summary": ""
                            }
                        
                        # Batting
                        batting_stats = stats.get("batting", {}).get("statistics")
                        if batting_stats:
                            player_stats[player_id]["stats"]["batting"] = {
                                "at_bats": batting_stats.get("atBats", 0),
                                "hits": batting_stats.get("hits", 0),
                                "home_runs": batting_stats.get("homeRuns", 0),
                                "rbis": batting_stats.get("rbi", 0),
                                "average": batting_stats.get("avg", "0.000")
                            }
                            # Simple summary for now, can be more sophisticated
                            player_stats[player_id]["summary"] += f"{batting_stats.get('hits',0)}-{batting_stats.get('atBats',0)}, "
                            if batting_stats.get('homeRuns',0) > 0:
                                player_stats[player_id]["summary"] += f"{batting_stats.get('homeRuns',0)} HR, "
                            player_stats[player_id]["summary"] += f"{batting_stats.get('rbi',0)} RBI. "

                        # Pitching
                        pitching_stats = stats.get("pitching", {}).get("statistics")
                        if pitching_stats:
                            player_stats[player_id]["stats"]["pitching"] = {
                                "innings_pitched": pitching_stats.get("inningsPitched", "0.0"),
                                "hits": pitching_stats.get("hits", 0),
                                "runs": pitching_stats.get("runs", 0),
                                "earned_runs": pitching_stats.get("earnedRuns", 0),
                                "strikeouts": pitching_stats.get("strikeouts", 0),
                                "walks": pitching_stats.get("walks", 0),
                                "era": pitching_stats.get("era", "0.00")
                            }
                            # Simple summary for pitching
                            player_stats[player_id]["summary"] += f"{pitching_stats.get('inningsPitched','0.0')} IP, "
                            player_stats[player_id]["summary"] += f"{pitching_stats.get('earnedRuns',0)} ER, "
                            player_stats[player_id]["summary"] += f"{pitching_stats.get('strikeouts',0)} K. "
                            
                        player_stats[player_id]["summary"] = player_stats[player_id]["summary"].strip().strip(',')

        # Heuristic for top performers:
        # Prioritize players with HRs, high RBIs, good pitching lines (many IP, K, low ER)
        sorted_performers = sorted(
            player_stats.values(), 
            key=lambda p: (
                p.get("stats", {}).get("batting", {}).get("home_runs", 0) * 100 +
                p.get("stats", {}).get("batting", {}).get("rbis", 0) * 10 +
                p.get("stats", {}).get("pitching", {}).get("strikeouts", 0) * 5 +
                p.get("stats", {}).get("pitching", {}).get("innings_pitched", 0.0) * 2 -
                p.get("stats", {}).get("pitching", {}).get("earned_runs", 0) * 5
            ),
            reverse=True
        )
        
        return sorted_performers[:3] # Return top 3 performers

    def _extract_game_highlights(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract game highlights and context from detailed game data.
        This will be largely heuristic based on available data.
        """
        highlights = {
            "turning_point_inning": None,
            "win_probability_swing": 0.0,
            "total_runs": 0,
            "extra_innings": False
        }

        if not game_data or "liveData" not in game_data:
            return highlights
        
        live_data = game_data["liveData"]

        # Total runs
        home_runs = live_data.get("linescore", {}).get("teams", {}).get("home", {}).get("runs", 0)
        away_runs = live_data.get("linescore", {}).get("teams", {}).get("away", {}).get("runs", 0)
        highlights["total_runs"] = home_runs + away_runs

        # Extra innings
        current_inning = live_data.get("linescore", {}).get("currentInning", 0)
        highlights["extra_innings"] = current_inning > 9

        # Win probability swing - requires parsing win probability data if available
        # MLB-StatsAPI 'game' endpoint typically doesn't give direct WP swings
        # This would require more advanced parsing or another data source.
        # Placeholder for now.
        # For a full implementation, you might need to query `statsapi.get("winProbability")` if available
        # or calculate from play-by-play data.
        
        # Turning point inning - very complex to determine programmatically without advanced analytics
        # Could be tied to a key play with highest impact or biggest score swing
        
        return highlights

    def _extract_season_context(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract season context (team records, playoff implications) from detailed game data.
        This will mostly rely on pre-game data if available, or external lookups.
        """
        season_context = {
            "home_record": "N/A",
            "away_record": "N/A",
            "playoff_implications": "N/A" # This is very hard to determine from API alone
        }

        if not game_data or "gameData" not in game_data:
            return season_context
        
        game_data_section = game_data["gameData"]
        home_team_record = game_data_section.get("teams", {}).get("home", {}).get("record", {})
        away_team_record = game_data_section.get("teams", {}).get("away", {}).get("record", {})

        if home_team_record:
            season_context["home_record"] = f"{home_team_record.get('wins',0)}-{home_team_record.get('losses',0)}"
        if away_team_record:
            season_context["away_record"] = f"{away_team_record.get('wins',0)}-{away_team_record.get('losses',0)}"

        # Playoff implications would likely require external data or a more complex model
        return season_context