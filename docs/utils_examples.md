# MLB Video Pipeline - Utilities Usage Examples

This document provides examples for all core utilities in `src/utils/`.

## Table of Contents

1. [Logging System](#logging-system)
2. [Data Validators](#data-validators)
3. [Cost Tracker](#cost-tracker)
4. [Configuration](#configuration)
5. [Exception Handling](#exception-handling)

---

## Logging System

### Basic Usage

```python
from src.utils import get_logger

# Create a logger for your module
logger = get_logger(__name__)

# Log messages at different levels
logger.debug("Detailed debugging info")
logger.info("Fetching MLB games for 2024-07-04")
logger.warning("API response was slow")
logger.error("Failed to fetch game data", extra={"game_id": "12345"})
```

### With Context

```python
from src.utils import get_logger, log_context

logger = get_logger(__name__)

# Add context fields to all logs within a block
with log_context(logger, game_id="748589", date="2024-07-04"):
    logger.info("Processing game")
    logger.info("Generating script")
    logger.info("Game processed")  # All include game_id and date
```

### Log Files

Logs are stored in `logs/` directory:
- `pipeline.log` - Main application logs (JSON format, rotating)
- `costs.jsonl` - API cost tracking logs (JSON Lines format)

---

## Data Validators

### Game Data Validation

```python
from src.utils import validate_game_data

game = {
    "game_id": "748589",
    "date": "2024-07-04",
    "home_team": "New York Yankees",
    "away_team": "Boston Red Sox",
    "home_score": 5,
    "away_score": 2,
    "highlights": "https://example.com/highlights.mp4"  # Optional
}

is_valid, errors = validate_game_data(game)

if not is_valid:
    for error in errors:
        print(f"Validation error: {error}")
```

### Script Validation

```python
from src.utils import validate_script

script = """
Welcome to today's MLB highlights! The Yankees faced off against
the Red Sox in an exciting matchup at Yankee Stadium...
"""

is_valid, error = validate_script(script)

if not is_valid:
    print(f"Script error: {error}")
```

### Audio/Video Validation

```python
from src.utils import validate_audio, validate_video

# Validate audio file
is_valid, error = validate_audio(
    "/path/to/narration.mp3",
    min_duration=5.0,
    max_duration=180.0
)

# Validate video file
is_valid, error = validate_video(
    "/path/to/output.mp4",
    min_duration=30.0,
    min_width=1080,
    min_height=1920,
    max_file_size_mb=500
)
```

### Filename Sanitization

```python
from src.utils import sanitize_filename

# Sanitize user-provided filenames
safe_name = sanitize_filename("Game: Yankees vs Red Sox (2024)")
# Result: "Game_Yankees_vs_Red_Sox_2024"
```

---

## Cost Tracker

### Logging API Calls

```python
from src.utils import CostTracker, get_cost_tracker

# Get singleton instance
tracker = get_cost_tracker()

# Log OpenAI API call
cost = tracker.log_openai_call(
    input_tokens=500,
    output_tokens=200,
    model="gpt-4o"
)
print(f"Call cost: ${cost:.4f}")

# Log ElevenLabs API call
cost = tracker.log_elevenlabs_call(
    characters=450,
    model="standard"
)
```

### Checking Costs

```python
from src.utils import get_cost_tracker

tracker = get_cost_tracker()

# Get current totals
daily_total = tracker.get_daily_total()
monthly_total = tracker.get_monthly_total()

print(f"Today: ${daily_total:.4f}")
print(f"This month: ${monthly_total:.4f}")

# Get detailed summary
summary = tracker.get_daily_summary()
print(f"OpenAI: ${summary['breakdown']['openai']:.4f}")
print(f"ElevenLabs: ${summary['breakdown']['elevenlabs']:.4f}")
print(f"Remaining budget: ${summary['budget']['remaining']:.4f}")

# Check if within budget
if tracker.is_within_budget(estimated_cost=0.50):
    # Proceed with API call
    pass
```

### Budget Alerts

```python
from src.utils import CostTracker

def budget_alert(alert_type: str, current: float, limit: float):
    """Custom alert handler."""
    print(f"ALERT: {alert_type} budget at ${current:.2f} / ${limit:.2f}")
    # Could send email, Slack notification, etc.

tracker = CostTracker(
    daily_limit=10.0,
    monthly_limit=100.0,
    alert_callback=budget_alert
)
```

### Estimating Costs

```python
from src.utils import estimate_openai_cost, estimate_elevenlabs_cost

# Estimate before making API call
estimated = estimate_openai_cost(
    input_tokens=1000,
    output_tokens=500,
    model="gpt-4o"
)
print(f"Estimated cost: ${estimated:.4f}")

# Estimate TTS cost
estimated = estimate_elevenlabs_cost(characters=2000)
print(f"Estimated TTS cost: ${estimated:.4f}")
```

---

## Configuration

### Accessing Settings

```python
from src.utils import get_config, require_key, require_keys

# Get full config object
config = get_config()
print(config.openai_api_key)
print(config.daily_cost_limit)

# Require specific keys (raises ConfigurationError if missing)
api_key = require_key("openai_api_key")

# Require multiple keys
keys = require_keys(["openai_api_key", "elevenlabs_api_key"])
```

### Environment Detection

```python
from src.utils import is_production, is_development, is_testing

if is_production():
    # Use production settings
    pass
elif is_development():
    # Use development settings
    pass
```

### Feature Flags

```python
from src.utils import is_feature_enabled, is_dry_run

if is_feature_enabled("cost_tracking"):
    tracker.log_openai_call(...)

if is_dry_run():
    print("DRY RUN: Would call OpenAI API")
else:
    # Make actual API call
    pass
```

### Path Utilities

```python
from src.utils import (
    get_project_root,
    get_data_dir,
    get_logs_dir,
    get_outputs_dir,
    ensure_directories
)

# Get paths
project_root = get_project_root()
data_dir = get_data_dir()

# Create all required directories
ensure_directories()
```

### API Configuration

```python
from src.utils import get_api_config, validate_api_keys

# Get config for specific service
openai_config = get_api_config("openai")
print(openai_config["api_key"])
print(openai_config["max_tokens"])

# Check which API keys are configured
status = validate_api_keys(["openai", "elevenlabs"])
if not status["openai"]:
    print("Warning: OpenAI API key not configured")
```

---

## Exception Handling

### Using Custom Exceptions

```python
from src.utils import (
    DataFetchError,
    ScriptGenerationError,
    VideoGenerationError,
    wrap_exception
)

# Raise with context
def fetch_game(game_id: str):
    try:
        response = api.get(f"/games/{game_id}")
        if response.status_code != 200:
            raise DataFetchError(
                "Failed to fetch game data",
                source="MLB-StatsAPI",
                game_id=game_id,
                status_code=response.status_code
            )
        return response.json()
    except ConnectionError as e:
        raise wrap_exception(e, DataFetchError, source="MLB-API")
```

### Exception Context

```python
from src.utils import ScriptGenerationError

try:
    # Generate script
    pass
except Exception as e:
    error = ScriptGenerationError(
        "Failed to generate script",
        model="gpt-4o",
        game_id="12345",
        reason="content_filter"
    )

    # Log with full context
    print(error.to_dict())
    # {
    #     "error_type": "ScriptGenerationError",
    #     "error_code": "SCRIPT_ERROR",
    #     "message": "Failed to generate script",
    #     "timestamp": "2024-07-04T12:00:00Z",
    #     "context": {
    #         "model": "gpt-4o",
    #         "game_id": "12345",
    #         "reason": "content_filter"
    #     }
    # }
```

### Complete Error Handling Example

```python
from src.utils import (
    get_logger,
    DataFetchError,
    ScriptGenerationError,
    VideoGenerationError,
    BudgetExceededError,
    get_cost_tracker
)

logger = get_logger(__name__)
tracker = get_cost_tracker()

def generate_video(game_id: str):
    try:
        # Check budget first
        if not tracker.is_within_budget(estimated_cost=1.0):
            raise BudgetExceededError(
                "Insufficient budget for video generation",
                budget_type="daily",
                current=tracker.get_daily_total(),
                limit=tracker.daily_limit
            )

        # Fetch data
        game_data = fetch_game(game_id)

        # Generate script
        script = generate_script(game_data)

        # Generate video
        video = render_video(script)

        return video

    except DataFetchError as e:
        logger.error("Data fetch failed", extra=e.to_dict())
        raise

    except ScriptGenerationError as e:
        logger.error("Script generation failed", extra=e.to_dict())
        raise

    except VideoGenerationError as e:
        logger.error("Video generation failed", extra=e.to_dict())
        raise
```

---

## Quick Reference

### Imports

```python
# All-in-one import
from src.utils import (
    # Logging
    get_logger, log_context,

    # Validation
    validate_game_data, validate_script, validate_audio, validate_video,
    sanitize_filename,

    # Cost tracking
    CostTracker, get_cost_tracker, estimate_openai_cost,

    # Config
    get_config, require_key, is_production, is_dry_run,
    get_project_root, ensure_directories,

    # Exceptions
    DataFetchError, ScriptGenerationError, VideoGenerationError,
    ConfigurationError, wrap_exception,
)
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level | `INFO` |
| `ENVIRONMENT` | Environment name | `development` |
| `DRY_RUN` | Skip actual API calls | `false` |
| `DAILY_COST_LIMIT` | Daily budget USD | `10.0` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ELEVENLABS_API_KEY` | ElevenLabs API key | - |
