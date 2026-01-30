# Core Utilities: Usage Guide

This guide provides practical examples for using the core utilities located in the `src/utils/` directory.

## 1. Logging System (`src/utils/logger.py`)

The logging system provides a centralized way to record events. It logs to both the console (with colors) and a rotating file (`logs/pipeline.log`) in JSON format.

### Basic Usage

To get a logger instance for your module, use `get_logger(__name__)`.

```python
# In any module, e.g., src/data/fetcher.py
from src.utils.logger import get_logger
from src.utils.exceptions import DataFetchError

# Get a logger scoped to the current module
logger = get_logger(__name__)

def fetch_game_data(game_id: str):
    logger.info(f"Attempting to fetch data for game_id: {game_id}")
    try:
        # ... API call logic ...
        if response.status_code != 200:
            raise DataFetchError(
                message=f"API returned status {response.status_code}",
                source="MLB-StatsAPI",
                game_id=game_id
            )
        logger.info(f"Successfully fetched data for game_id: {game_id}")
        return response.json()
    except Exception as e:
        # Log the error with contextual information
        logger.error(
            "Failed to fetch game data",
            extra={"game_id": game_id, "error": str(e)},
            exc_info=True  # This adds traceback information
        )
        # Re-raise or handle the exception
        raise

fetch_game_data("748589")
```

### Log Levels

The default log level is `INFO`. You can change this by setting the `LOG_LEVEL` environment variable in your `.env` file.

- `DEBUG`: Detailed information, typically of interest only when diagnosing problems.
- `INFO`: Confirmation that things are working as expected.
- `WARNING`: An indication that something unexpected happened, or indicative of some problem in the near future.
- `ERROR`: Due to a more serious problem, the software has not been able to perform some function.
- `CRITICAL`: A very serious error, indicating that the program itself may be unable to continue running.

## 2. Configuration (`src/utils/config.py`)

All project-wide settings are managed in `src/utils/config.py` and loaded from your `.env` file.

### Accessing Settings

Import the `settings` object from `src.utils.config`.

```python
from src.utils.config import get_settings

# Get the settings instance
settings = get_settings()

# Access any setting
print(f"Using OpenAI model: {settings.SCRIPT_GENERATION_MODEL}")
print(f"Output directory is: {settings.OUTPUT_DIR}")

# The API keys are loaded automatically from your .env file
openai_key = settings.OPENAI_API_KEY
```

To add a new setting, simply add it to the `Settings` class in `src/utils/config.py` and, if it's not a secret, provide a default value. If it's a secret (like an API key), add it to your `.env` and `.env.example` files.

## 3. Data Validators (`src/utils/validators.py`)

Validators ensure that data structures conform to their expected schemas before being processed.

### Validating Game Data

This is useful after fetching data from an API.

```python
from src.utils.validators import validate_game_data

# Example of data that is missing a field ('home_score')
bad_game_data = {
    "game_id": "748589",
    "date": "2024-07-21",
    "home_team": "New York Yankees",
    "away_team": "Boston Red Sox",
    # "home_score": 5, # Missing
    "away_score": 2,
}

is_valid, errors = validate_game_data(bad_game_data)

if not is_valid:
    print(f"Game data is invalid: {errors}")
    # Output: Game data is invalid: ["home_score: Field required"]
```

## 4. Cost Tracker (`src/utils/cost_tracker.py`)

The cost tracker provides a way to monitor estimated API costs. It's a singleton, so you can call it from anywhere, and it will maintain a consistent state.

```python
from src.utils.cost_tracker import CostTracker

# Get the singleton instance
tracker = CostTracker()

# Log an OpenAI call
tracker.log_openai_call(
    model="gpt-4o",
    input_tokens=1500,
    output_tokens=4000
)

# Log an ElevenLabs call
tracker.log_elevenlabs_call(characters=2500)

# Get current totals
daily_cost = tracker.get_daily_total()
monthly_cost = tracker.get_monthly_total()

print(f"Estimated cost today: ${daily_cost:.4f}")
print(f"Estimated cost this month: ${monthly_cost:.4f}")
```

Costs are automatically logged to `logs/costs.json`.

## 5. Custom Exceptions (`src/utils/exceptions.py`)

Use custom exceptions to provide specific, context-rich error information. This makes debugging much easier.

```python
from src.utils.exceptions import ScriptGenerationError, VideoGenerationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

def generate_script(game_data: dict):
    try:
        # ... logic to call OpenAI ...
        if "error" in response_from_openai:
            raise ScriptGenerationError(
                "OpenAI API returned an error",
                model="gpt-4o",
                game_id=game_data.get("game_id"),
                api_error=response_from_openai["error"]
            )
        return response_from_openai["script"]
    except Exception as e:
        logger.error("An unexpected error occurred during script generation", exc_info=True)
        # Wrap the original exception if it's not already a custom one
        if not isinstance(e, ScriptGenerationError):
            raise ScriptGenerationError(str(e), model="gpt-4o", game_id=game_data.get("game_id")) from e
        raise

def render_video(script: str, assets: list):
    try:
        # ... moviepy logic ...
        if something_goes_wrong:
            raise VideoGenerationError(
                "Failed to compile clips",
                step="timeline_assembly",
                clip_count=len(assets)
            )
    except Exception as e:
        logger.error(f"Video rendering failed: {e}", exc_info=True)
        raise
```
