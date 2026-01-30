import pytest
from unittest.mock import patch, MagicMock
import os
import json
import datetime
from datetime import timedelta
import time
import requests

from src.data.fetcher import MLBDataFetcher, DataFetchError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Helper function to create dummy game data
def create_dummy_game_data(game_id: int, status_code: str, date_str: str = "2024-07-04") -> dict:
    return {
        "gamePk": game_id,
        "game_id": game_id,
        "game_date": date_str,
        "status": {"statusCode": status_code, "detailedState": "Final" if status_code == "F" else "Scheduled"},
        "home_team": {"name": "Home Team", "abbreviation": "HT"},
        "away_team": {"name": "Away Team", "abbreviation": "AT"},
        "home_score": 5 if status_code == "F" else 0,
        "away_score": 3 if status_code == "F" else 0,
        "venue": {"name": "Stadium"},
        "liveData": { # Simplified liveData for full detail
            "plays": {"allPlays": []},
            "boxscore": {"teams": []},
            "linescore": {"teams": {"home": {"runs": 5}, "away": {"runs": 3}}}
        },
        "gameData": { # Simplified gameData for full detail
            "datetime": {"officialDate": date_str},
            "status": {"statusCode": status_code, "detailedState": "Final" if status_code == "F" else "Scheduled"},
            "teams": {
                "home": {"name": "Home Team", "abbreviation": "HT", "record": {"wins": 10, "losses": 5}},
                "away": {"name": "Away Team", "abbreviation": "AT", "record": {"wins": 8, "losses": 7}},
            },
            "venue": {"name": "Stadium", "location": {"cityState": "Some City, ST"}}
        }
    }

@pytest.fixture
def fetcher_instance(tmp_path):
    """Fixture for MLBDataFetcher with a temporary cache directory."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return MLBDataFetcher(cache_dir=str(cache_dir))

class TestMLBDataFetcher:

    @patch('statsapi.schedule')
    @patch('statsapi.get')
    def test_fetch_games_no_games(self, mock_statsapi_get, mock_statsapi_schedule, fetcher_instance):
        mock_statsapi_schedule.return_value = []
        games = fetcher_instance.fetch_games("2024-01-01")
        assert len(games) == 0
        mock_statsapi_schedule.assert_called_once()
        mock_statsapi_get.assert_not_called()

    @patch('statsapi.schedule')
    @patch('statsapi.get')
    def test_fetch_games_success(self, mock_statsapi_get, mock_statsapi_schedule, fetcher_instance):
        game_id_1 = 12345
        game_id_2 = 67890
        date_str = "2024-07-04"

        # Mock schedule to return two game summaries
        mock_statsapi_schedule.return_value = [
            {"game_id": game_id_1, "game_date": date_str, "status": {"statusCode": "S"}},
            {"game_id": game_id_2, "game_date": date_str, "status": {"statusCode": "S"}}
        ]
        
        # Mock get for game details
        mock_statsapi_get.side_effect = [
            create_dummy_game_data(game_id_1, "F", date_str),
            create_dummy_game_data(game_id_2, "F", date_str)
        ]

        games = fetcher_instance.fetch_games(date_str)

        assert len(games) == 2
        assert games[0]["game_id"] == game_id_1
        assert games[1]["game_id"] == game_id_2
        mock_statsapi_schedule.assert_called_once_with(date=datetime.date(2024, 7, 4))
        assert mock_statsapi_get.call_count == 2
        mock_statsapi_get.assert_any_call(endpoint='game', gamePk=game_id_1)
        mock_statsapi_get.assert_any_call(endpoint='game', gamePk=game_id_2)

        # Check caching
        cache_file_1 = os.path.join(fetcher_instance.cache_dir, f"game_detail_{game_id_1}.json")
        cache_file_2 = os.path.join(fetcher_instance.cache_dir, f"game_detail_{game_id_2}.json")
        assert os.path.exists(cache_file_1)
        assert os.path.exists(cache_file_2)
        
        schedule_cache_file = os.path.join(fetcher_instance.cache_dir, f"schedule_{date_str}.json")
        assert os.path.exists(schedule_cache_file)


    def test_fetch_games_invalid_date_format(self, fetcher_instance):
        with pytest.raises(DataFetchError, match="Invalid date format"):
            fetcher_instance.fetch_games("2024/01/01")

    @patch('statsapi.get')
    def test_get_game_details_success(self, mock_statsapi_get, fetcher_instance):
        game_id = 746969
        dummy_data = create_dummy_game_data(game_id, "F")
        mock_statsapi_get.return_value = dummy_data

        details = fetcher_instance.get_game_details(game_id)
        assert details is not None
        assert details["game_id"] == game_id
        mock_statsapi_get.assert_called_once_with(endpoint='game', gamePk=game_id)

        # Check cache
        game_date_str = dummy_data["gameData"]["datetime"]["officialDate"]
        cache_file = os.path.join(fetcher_instance.cache_dir, f"game_detail_{game_id}.json")
        assert os.path.exists(cache_file)
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
            assert cached_data["game_id"] == game_id

    @patch('statsapi.get')
    def test_get_game_details_cache_hit(self, mock_statsapi_get, fetcher_instance):
        game_id = 746969
        date_str = "2024-07-04"
        cached_data = create_dummy_game_data(game_id, "F", date_str)
        # Manually save to cache
        cache_file = os.path.join(fetcher_instance.cache_dir, f"game_detail_{game_id}.json")
        with open(cache_file, 'w') as f:
            json.dump(cached_data, f)
        
        details = fetcher_instance.get_game_details(game_id)
        assert details["game_id"] == game_id
        mock_statsapi_get.assert_not_called() # Should not call API if cached

    @patch('statsapi.get')
    def test_get_game_details_cache_stale_non_final(self, mock_statsapi_get, fetcher_instance):
        game_id = 746969
        date_str = "2024-07-04"
        
        # Create a cached non-final game (e.g., In Progress)
        cached_data = create_dummy_game_data(game_id, "I", date_str)
        cached_data["cache_timestamp"] = (datetime.datetime.now() - timedelta(minutes=10)).isoformat() # Older than 5 min
        
        cache_file = os.path.join(fetcher_instance.cache_dir, f"game_detail_{game_id}.json")
        with open(cache_file, 'w') as f:
            json.dump(cached_data, f)
        
        # API call should happen
        mock_statsapi_get.return_value = create_dummy_game_data(game_id, "F", date_str)
        
        details = fetcher_instance.get_game_details(game_id)
        assert details["game_id"] == game_id
        mock_statsapi_get.assert_called_once() # API should be called

    @patch('statsapi.schedule')
    @patch('statsapi.get')
    def test_fetch_games_schedule_cache_stale(self, mock_statsapi_get, mock_statsapi_schedule, fetcher_instance):
        date_str = "2024-07-04"
        game_id_1 = 111
        game_id_2 = 222

        # Create a stale schedule cache with an in-progress game
        stale_schedule = [
            {"game_id": game_id_1, "game_date": date_str, "status": {"statusCode": "I"}},
            {"game_id": game_id_2, "game_date": date_str, "status": {"statusCode": "S"}}
        ]
        fetcher_instance._save_to_cache({"games": stale_schedule, "cache_timestamp": (datetime.datetime.now() - timedelta(hours=2)).isoformat()}, date_str)

        # Mock API calls when cache is considered stale
        mock_statsapi_schedule.return_value = [
            {"game_id": game_id_1, "game_date": date_str, "status": {"statusCode": "F"}},
            {"game_id": game_id_2, "game_date": date_str, "status": {"statusCode": "F"}}
        ]
        mock_statsapi_get.side_effect = [
            create_dummy_game_data(game_id_1, "F", date_str),
            create_dummy_game_data(game_id_2, "F", date_str)
        ]

        games = fetcher_instance.fetch_games(date_str)
        assert len(games) == 2
        mock_statsapi_schedule.assert_called_once()
        assert mock_statsapi_get.call_count == 2
        
        # Verify old cache is removed (or overwritten)
        schedule_cache_file = os.path.join(fetcher_instance.cache_dir, f"schedule_{date_str}.json")
        with open(schedule_cache_file, 'r') as f:
            updated_cache = json.load(f)
            assert updated_cache.get("cache_timestamp") is not None
            assert (datetime.datetime.now() - datetime.datetime.fromisoformat(updated_cache["cache_timestamp"])) < timedelta(minutes=1)

    @patch('statsapi.schedule')
    @patch('statsapi.get')
    def test_fetch_games_schedule_cache_valid_final_games(self, mock_statsapi_get, mock_statsapi_schedule, fetcher_instance):
        date_str = "2024-07-04"
        game_id_1 = 333
        game_id_2 = 444

        # Create a valid schedule cache with final games
        valid_schedule_games = [
            create_dummy_game_data(game_id_1, "F", date_str),
            create_dummy_game_data(game_id_2, "F", date_str)
        ]
        # fetcher_instance._save_to_cache({"games": valid_schedule_games, "cache_timestamp": datetime.datetime.now().isoformat()}, date_str)

        # Manually save to cache, bypassing _save_to_cache's timestamp logic for this test
        schedule_cache_file = os.path.join(fetcher_instance.cache_dir, f"schedule_{date_str}.json")
        with open(schedule_cache_file, 'w') as f:
            json.dump({"games": valid_schedule_games, "cache_timestamp": datetime.datetime.now().isoformat()}, f)
        
        # Also cache individual game details as fetch_games would do
        fetcher_instance._save_to_cache(valid_schedule_games[0], date_str, game_id_1)
        fetcher_instance._save_to_cache(valid_schedule_games[1], date_str, game_id_2)


        games = fetcher_instance.fetch_games(date_str)
        assert len(games) == 2
        mock_statsapi_schedule.assert_not_called() # Should not call API if schedule cache is valid
        mock_statsapi_get.assert_not_called() # Should not call API if game details are also valid and final

    @patch('statsapi.schedule')
    @patch('statsapi.get', side_effect=DataFetchError("API error", source="test"))
    def test_fetch_games_api_error_propagation(self, mock_statsapi_get, mock_statsapi_schedule, fetcher_instance):
        date_str = "2024-07-04"
        mock_statsapi_schedule.return_value = [{"game_id": 123, "game_date": date_str, "status": {"statusCode": "S"}}]
        
        with pytest.raises(DataFetchError):
            fetcher_instance.fetch_games(date_str)

    @patch('statsapi.get', side_effect=requests.exceptions.RequestException("Network Error"))
    def test_make_api_call_retries(self, mock_statsapi_get, fetcher_instance):
        with pytest.raises(DataFetchError):
            fetcher_instance._make_api_call(mock_statsapi_get, endpoint='game', gamePk=123)
        assert mock_statsapi_get.call_count == 3 # 1 initial + 2 retries

    def test_extract_key_plays(self, fetcher_instance):
        # A more detailed mock for game data for key plays
        game_data = {
            "gamePk": 123,
            "liveData": {
                "plays": {
                    "allPlays": [
                        { # Home run
                            "about": {"inning": 1},
                            "playEvents": [{
                                "details": {"event": "Home Run"},
                                "description": "Aaron Judge homers",
                            }],
                            "result": {"rbi": 1}
                        },
                        { # Strikeout
                            "about": {"inning": 8},
                            "playEvents": [{
                                "details": {"event": "Strikeout"},
                                "description": "Gerrit Cole strikes out batter",
                            }],
                            "result": {"rbi": 0} # No RBI, just a strikeout
                        },
                        { # RBI Double
                            "about": {"inning": 5},
                            "playEvents": [{
                                "details": {"event": "Double"},
                                "description": "Player hits a 2-run double",
                            }],
                            "result": {"rbi": 2}
                        }
                    ]
                }
            }
        }
        key_plays = fetcher_instance._extract_key_plays(game_data)
        assert len(key_plays) == 3
        assert any(p["type"] == "home_run" for p in key_plays)
        assert any(p["type"] == "strikeout" and p["impact"] == "medium" for p in key_plays) # Inning 8
        assert any(p["type"] == "rbi_double" and p["impact"] == "medium" for p in key_plays)

    def test_extract_top_performers(self, fetcher_instance):
        game_data = {
            "gamePk": 123,
            "liveData": {
                "boxscore": {
                    "teams": [
                        {
                            "team": {"id": 1, "abbreviation": "NYY"},
                            "players": {
                                "ID1": {
                                    "fullName": "Aaron Judge", "position": {"abbreviation": "RF"}, "status": {"description": "Active"},
                                    "stats": {"batting": {"statistics": {"atBats": 4, "hits": 2, "homeRuns": 2, "rbi": 3, "avg": ".321"}}}
                                },
                                "ID2": {
                                    "fullName": "Gerrit Cole", "position": {"abbreviation": "P"}, "status": {"description": "Active"},
                                    "stats": {"pitching": {"statistics": {"inningsPitched": 8.0, "hits": 4, "runs": 2, "earnedRuns": 2, "strikeouts": 10, "walks": 1, "era": "2.89"}}}
                                },
                                "ID3": {
                                    "fullName": "Luke Voit", "position": {"abbreviation": "1B"}, "status": {"description": "Active"},
                                    "stats": {"batting": {"statistics": {"atBats": 3, "hits": 1, "homeRuns": 0, "rbi": 0, "avg": ".250"}}}
                                },
                            }
                        }
                    ]
                }
            }
        }
        performers = fetcher_instance._extract_top_performers(game_data)
        assert len(performers) == 3
        assert performers[0]["name"] == "Aaron Judge"
        assert performers[1]["name"] == "Gerrit Cole"
        assert performers[2]["name"] == "Luke Voit"
        assert "2-4, 2 HR, 3 RBI" in performers[0]["summary"]
        assert "8.0 IP, 2 ER, 10 K" in performers[1]["summary"]

    def test_extract_game_highlights(self, fetcher_instance):
        game_data = {
            "gamePk": 123,
            "liveData": {
                "linescore": {
                    "currentInning": 10, # Extra innings
                    "teams": {"home": {"runs": 6}, "away": {"runs": 5}}
                }
            }
        }
        highlights = fetcher_instance._extract_game_highlights(game_data)
        assert highlights["total_runs"] == 11
        assert highlights["extra_innings"] is True
        # Win probability swing and turning_point_inning are placeholders for now

    def test_extract_season_context(self, fetcher_instance):
        game_data = {
            "gamePk": 123,
            "gameData": {
                "teams": {
                    "home": {"record": {"wins": 52, "losses": 34}},
                    "away": {"record": {"wins": 45, "losses": 42}}
                }
            }
        }
        context = fetcher_instance._extract_season_context(game_data)
        assert context["home_record"] == "52-34"
        assert context["away_record"] == "45-42"
        assert context["playoff_implications"] == "N/A" # Placeholder

