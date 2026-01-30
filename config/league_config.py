"""
MLB Video Pipeline - League Configuration

MLB-specific constants, team data, and statistical categories.
This file contains all baseball-specific configuration that doesn't
change between environments.

Usage:
    from config.league_config import MLB_TEAMS, STAT_CATEGORIES, get_team_info
"""

from typing import TypedDict


# =============================================================================
# Type Definitions
# =============================================================================

class TeamInfo(TypedDict):
    """Team information structure."""
    name: str
    abbreviation: str
    league: str
    division: str
    city: str
    colors: tuple[str, str]  # Primary, secondary


class StatCategory(TypedDict):
    """Statistical category definition."""
    name: str
    abbreviation: str
    description: str
    higher_is_better: bool
    format: str  # e.g., ".3f" for batting average


# =============================================================================
# MLB Teams
# =============================================================================

MLB_TEAMS_BY_ID: dict[int, TeamInfo] = {
    # American League East
    110: {"name": "Orioles", "abbreviation": "BAL", "league": "AL", "division": "East",
          "city": "Baltimore", "colors": ("#DF4601", "#000000")},
    111: {"name": "Red Sox", "abbreviation": "BOS", "league": "AL", "division": "East",
          "city": "Boston", "colors": ("#BD3039", "#0C2340")},
    147: {"name": "Yankees", "abbreviation": "NYY", "league": "AL", "division": "East",
          "city": "New York", "colors": ("#003087", "#E4002C")},
    139: {"name": "Rays", "abbreviation": "TB", "league": "AL", "division": "East",
          "city": "Tampa Bay", "colors": ("#092C5C", "#8FBCE6")},
    141: {"name": "Blue Jays", "abbreviation": "TOR", "league": "AL", "division": "East",
          "city": "Toronto", "colors": ("#134A8E", "#1D2D5C")},

    # American League Central
    118: {"name": "Royals", "abbreviation": "KC", "league": "AL", "division": "Central",
          "city": "Kansas City", "colors": ("#004687", "#BD9B60")},
    114: {"name": "Guardians", "abbreviation": "CLE", "league": "AL", "division": "Central",
          "city": "Cleveland", "colors": ("#00385D", "#E50022")},
    116: {"name": "Tigers", "abbreviation": "DET", "league": "AL", "division": "Central",
          "city": "Detroit", "colors": ("#0C2340", "#FA4616")},
    142: {"name": "Twins", "abbreviation": "MIN", "league": "AL", "division": "Central",
          "city": "Minnesota", "colors": ("#002B5C", "#D31145")},
    145: {"name": "White Sox", "abbreviation": "CWS", "league": "AL", "division": "Central",
          "city": "Chicago", "colors": ("#27251F", "#C4CED4")},

    # American League West
    117: {"name": "Astros", "abbreviation": "HOU", "league": "AL", "division": "West",
          "city": "Houston", "colors": ("#002D62", "#EB6E1F")},
    108: {"name": "Angels", "abbreviation": "LAA", "league": "AL", "division": "West",
          "city": "Los Angeles", "colors": ("#BA0021", "#003263")},
    133: {"name": "Athletics", "abbreviation": "OAK", "league": "AL", "division": "West",
          "city": "Oakland", "colors": ("#003831", "#EFB21E")},
    136: {"name": "Mariners", "abbreviation": "SEA", "league": "AL", "division": "West",
          "city": "Seattle", "colors": ("#0C2C56", "#005C5C")},
    140: {"name": "Rangers", "abbreviation": "TEX", "league": "AL", "division": "West",
          "city": "Texas", "colors": ("#003278", "#C0111F")},

    # National League East
    144: {"name": "Braves", "abbreviation": "ATL", "league": "NL", "division": "East",
          "city": "Atlanta", "colors": ("#CE1141", "#13274F")},
    146: {"name": "Marlins", "abbreviation": "MIA", "league": "NL", "division": "East",
          "city": "Miami", "colors": ("#00A3E0", "#EF3340")},
    121: {"name": "Mets", "abbreviation": "NYM", "league": "NL", "division": "East",
          "city": "New York", "colors": ("#002D72", "#FF5910")},
    143: {"name": "Phillies", "abbreviation": "PHI", "league": "NL", "division": "East",
          "city": "Philadelphia", "colors": ("#E81828", "#002D72")},
    120: {"name": "Nationals", "abbreviation": "WSH", "league": "NL", "division": "East",
          "city": "Washington", "colors": ("#AB0003", "#14225A")},

    # National League Central
    112: {"name": "Cubs", "abbreviation": "CHC", "league": "NL", "division": "Central",
          "city": "Chicago", "colors": ("#0E3386", "#CC3433")},
    113: {"name": "Reds", "abbreviation": "CIN", "league": "NL", "division": "Central",
          "city": "Cincinnati", "colors": ("#C6011F", "#000000")},
    158: {"name": "Brewers", "abbreviation": "MIL", "league": "NL", "division": "Central",
          "city": "Milwaukee", "colors": ("#12284B", "#B6922E")},
    134: {"name": "Pirates", "abbreviation": "PIT", "league": "NL", "division": "Central",
          "city": "Pittsburgh", "colors": ("#27251F", "#FDB827")},
    138: {"name": "Cardinals", "abbreviation": "STL", "league": "NL", "division": "Central",
          "city": "St. Louis", "colors": ("#C41E3A", "#0C2340")},

    # National League West
    109: {"name": "Diamondbacks", "abbreviation": "ARI", "league": "NL", "division": "West",
          "city": "Arizona", "colors": ("#A71930", "#E3D4AD")},
    115: {"name": "Rockies", "abbreviation": "COL", "league": "NL", "division": "West",
          "city": "Colorado", "colors": ("#333366", "#131413")},
    119: {"name": "Dodgers", "abbreviation": "LAD", "league": "NL", "division": "West",
          "city": "Los Angeles", "colors": ("#005A9C", "#EF3E42")},
    135: {"name": "Padres", "abbreviation": "SD", "league": "NL", "division": "West",
          "city": "San Diego", "colors": ("#2F241D", "#FFC425")},
    137: {"name": "Giants", "abbreviation": "SF", "league": "NL", "division": "West",
          "city": "San Francisco", "colors": ("#FD5A1E", "#27251F")},
}

MLB_TEAMS_BY_ABBR: dict[str, TeamInfo] = {
    team_info["abbreviation"].upper(): team_info
    for team_id, team_info in MLB_TEAMS_BY_ID.items()
}

# Alias for backwards compatibility
MLB_TEAMS = MLB_TEAMS_BY_ID



# =============================================================================
# Statistical Categories
# =============================================================================

BATTING_STATS: dict[str, StatCategory] = {
    "avg": {"name": "Batting Average", "abbreviation": "AVG",
            "description": "Hits divided by at-bats", "higher_is_better": True, "format": ".3f"},
    "obp": {"name": "On-Base Percentage", "abbreviation": "OBP",
            "description": "Times reached base per plate appearance", "higher_is_better": True, "format": ".3f"},
    "slg": {"name": "Slugging Percentage", "abbreviation": "SLG",
            "description": "Total bases divided by at-bats", "higher_is_better": True, "format": ".3f"},
    "ops": {"name": "On-Base Plus Slugging", "abbreviation": "OPS",
            "description": "OBP + SLG", "higher_is_better": True, "format": ".3f"},
    "hr": {"name": "Home Runs", "abbreviation": "HR",
           "description": "Total home runs", "higher_is_better": True, "format": "d"},
    "rbi": {"name": "Runs Batted In", "abbreviation": "RBI",
            "description": "Runs driven in", "higher_is_better": True, "format": "d"},
    "sb": {"name": "Stolen Bases", "abbreviation": "SB",
           "description": "Successful stolen bases", "higher_is_better": True, "format": "d"},
    "war": {"name": "Wins Above Replacement", "abbreviation": "WAR",
            "description": "Total value over replacement player", "higher_is_better": True, "format": ".1f"},
}

PITCHING_STATS: dict[str, StatCategory] = {
    "era": {"name": "Earned Run Average", "abbreviation": "ERA",
            "description": "Earned runs per 9 innings", "higher_is_better": False, "format": ".2f"},
    "whip": {"name": "Walks + Hits per IP", "abbreviation": "WHIP",
             "description": "Walks and hits per inning pitched", "higher_is_better": False, "format": ".2f"},
    "k9": {"name": "Strikeouts per 9", "abbreviation": "K/9",
           "description": "Strikeouts per 9 innings", "higher_is_better": True, "format": ".1f"},
    "bb9": {"name": "Walks per 9", "abbreviation": "BB/9",
            "description": "Walks per 9 innings", "higher_is_better": False, "format": ".1f"},
    "wins": {"name": "Wins", "abbreviation": "W",
             "description": "Total wins", "higher_is_better": True, "format": "d"},
    "saves": {"name": "Saves", "abbreviation": "SV",
              "description": "Total saves", "higher_is_better": True, "format": "d"},
    "ip": {"name": "Innings Pitched", "abbreviation": "IP",
           "description": "Total innings pitched", "higher_is_better": True, "format": ".1f"},
}


# =============================================================================
# Positions
# =============================================================================

POSITIONS = {
    "P": "Pitcher",
    "C": "Catcher",
    "1B": "First Base",
    "2B": "Second Base",
    "3B": "Third Base",
    "SS": "Shortstop",
    "LF": "Left Field",
    "CF": "Center Field",
    "RF": "Right Field",
    "DH": "Designated Hitter",
    "PH": "Pinch Hitter",
    "PR": "Pinch Runner",
}


# =============================================================================
# Game States
# =============================================================================

GAME_STATES = {
    "Preview": "Scheduled",
    "Pre-Game": "Pre-Game",
    "Warmup": "Warmup",
    "In Progress": "Live",
    "Final": "Completed",
    "Game Over": "Completed",
    "Postponed": "Postponed",
    "Suspended": "Suspended",
    "Cancelled": "Cancelled",
}


# =============================================================================
# Video Content Types
# =============================================================================

VIDEO_CONTENT_TYPES = [
    "game_recap",           # Post-game summary
    "player_spotlight",     # Focus on one player
    "prediction",           # Pre-game predictions
    "weekly_roundup",       # Week in review
    "stat_breakdown",       # Deep dive into stats
    "matchup_preview",      # Series/game preview
]


# =============================================================================
# Helper Functions
# =============================================================================

def get_team_info(team_id: int) -> TeamInfo | None:
    """Get team information by MLB team ID."""
    return MLB_TEAMS_BY_ID.get(team_id)


def get_team_by_abbreviation(abbr: str) -> TeamInfo | None:
    """Get team information by abbreviation (e.g., 'NYY')."""
    return MLB_TEAMS_BY_ABBR.get(abbr.upper())


def get_teams_by_division(league: str, division: str) -> dict[int, TeamInfo]:
    """Get all teams in a specific division."""
    return {
        team_id: team_info
        for team_id, team_info in MLB_TEAMS_BY_ID.items()
        if team_info["league"] == league and team_info["division"] == division
    }


def format_stat(value: float, stat_key: str, stat_type: str = "batting") -> str:
    """Format a statistic value according to its category."""
    stats = BATTING_STATS if stat_type == "batting" else PITCHING_STATS
    if stat_key not in stats:
        return str(value)

    fmt = stats[stat_key]["format"]
    if fmt == "d":
        return str(int(value))
    return f"{value:{fmt}}"


# =============================================================================
# Series Configuration
# =============================================================================

SERIES_CONFIG = {
    "typical_length": 3,
    "max_length": 4,

    "video_types": {
        "middle": {
            "duration": 60,
            "parts": {"analysis": 30, "preview": 30}
        },
        "end": {
            "duration": 60,
            "parts": {"analysis": 20, "team1": 20, "team2": 20}
        }
    }
}


# =============================================================================
# Cross-Season Data Configuration
# =============================================================================

CROSS_SEASON_CONFIG = {
    "sequence_length": 10,          # Games per sequence
    "max_lookback_seasons": 2,      # Max seasons to look back
    "min_games_for_prediction": 5,  # Min games needed for prediction
    "include_minors": False,        # Whether to include minor league data
}


# =============================================================================
# Prediction Model Configuration
# =============================================================================

PREDICTION_CONFIG = {
    # 3-class classification labels
    "classes": {
        0: "below_average",
        1: "average",
        2: "above_average"
    },

    # Class thresholds (for interpreting predictions)
    "thresholds": {
        "below_average": {"max_ops": 0.650},
        "average": {"min_ops": 0.650, "max_ops": 0.800},
        "above_average": {"min_ops": 0.800}
    },

    # Weights for different reasoning categories
    "reasoning_types": {
        "recent_form": {"weight": 0.30, "description": "Last 10 games performance"},
        "matchup_history": {"weight": 0.25, "description": "vs this pitcher"},
        "home_advantage": {"weight": 0.15, "description": "Home/road splits"},
        "platoon": {"weight": 0.15, "description": "L/R advantage"},
        "pitch_type": {"weight": 0.10, "description": "Pitch type matchups"},
        "milestone": {"weight": 0.05, "description": "Approaching milestone"}
    },

    # Model architecture config
    "model": {
        "input_features": 11,       # Features per timestep
        "sequence_length": 10,      # Timesteps
        "hidden_dim": 64,           # LSTM hidden dimension
        "num_layers": 2,            # LSTM layers
        "num_classes": 3,           # Output classes
        "dropout": 0.3
    }
}
