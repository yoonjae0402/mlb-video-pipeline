# Getting Started Guide

This guide walks you through setting up and using the MLB Video Pipeline from scratch.

## Prerequisites

- Python 3.10+
- FFmpeg installed (`brew install ffmpeg` on macOS)
- API keys for OpenAI, ElevenLabs (optional: YouTube)

## Step 1: Initial Setup

### Clone and Install

```bash
cd mlb-video-pipeline

# Create virtual environment
python3 -m venv mlb-env
source mlb-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your editor
nano .env  # or vim, code, etc.
```

Add your API keys:

```
OPENAI_API_KEY=sk-your-key-here
ELEVENLABS_API_KEY=your-key-here
```

### Verify Setup

```bash
python scripts/test_pipeline.py
```

You should see all tests passing (some may skip if API keys aren't configured).

## Step 2: Fetch MLB Data

### Get Today's Games

```bash
python scripts/fetch_games.py
```

### Get Games for a Specific Date

```bash
python scripts/fetch_games.py --date 2024-07-04
```

### Get Games for a Team

```bash
python scripts/fetch_games.py --date 2024-07-04 --team NYY
```

### Get Games for a Date Range

```bash
python scripts/fetch_games.py --start 2024-07-01 --end 2024-07-07
```

Data is saved to `data/processed/`.

## Step 3: Generate Your First Video

### Dry Run (No API Calls)

Test the pipeline without spending API credits:

```bash
python scripts/generate_video.py --date 2024-07-04 --dry-run
```

### Real Generation

```bash
python scripts/generate_video.py --date 2024-07-04
```

This will:
1. Fetch game data for that date
2. Generate a script with GPT-4
3. Create narration with ElevenLabs
4. Produce a video with stats/graphics

Output is saved to `outputs/videos/`.

### Customize the Video

```bash
# Change template
python scripts/generate_video.py --date 2024-07-04 --template classic_baseball

# Skip audio (faster)
python scripts/generate_video.py --date 2024-07-04 --skip-audio

# Set duration
python scripts/generate_video.py --date 2024-07-04 --duration 90
```

## Step 4: Train the Prediction Model (Optional)

### Generate Training Data

First, collect historical game data:

```bash
# Fetch a month of games
python scripts/fetch_games.py --start 2024-06-01 --end 2024-06-30
```

### Train the Model

```bash
python scripts/train_model.py --epochs 50
```

The model is saved to `models/player_predictor.pth`.

## Step 5: Monitor with Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run dashboard/streamlit_app.py
```

Open http://localhost:8501 to see:
- Pipeline status
- Cost tracking
- Generated videos
- Data management

## Step 6: Upload to YouTube (Optional)

### Setup YouTube API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project
3. Enable YouTube Data API v3
4. Create OAuth 2.0 credentials
5. Download as `credentials.json` to project root

### Authenticate

```python
from src.upload import YouTubeUploader

uploader = YouTubeUploader()
uploader.authenticate()  # Opens browser for OAuth
```

### Upload a Video

```python
video_id = uploader.upload(
    video_path="outputs/videos/recap_12345.mp4",
    title="MLB Game Recap: Yankees vs Red Sox",
    description="AI-generated game analysis...",
    tags=["MLB", "baseball", "Yankees"],
    privacy="unlisted"  # or "public", "private"
)
```

## Common Tasks

### Check API Costs

```python
from src.utils.logger import CostTracker

tracker = CostTracker()
print(f"Today's spend: ${tracker.get_daily_total():.2f}")
print(tracker.get_summary())
```

### Clear Cache

```python
from src.data import MLBDataFetcher

fetcher = MLBDataFetcher()
fetcher.clear_cache()  # Clear all
fetcher.clear_cache(older_than_hours=24)  # Clear old only
```

### Use Different Video Templates

```python
from src.video import VideoGenerator
from src.video.templates import list_templates

# See available templates
for t in list_templates():
    print(f"{t['name']}: {t['description']}")

# Use a template
generator = VideoGenerator(template="electric_neon")
```

### Create Custom Scenes

```python
from src.video import VideoGenerator

generator = VideoGenerator()

scenes = [
    {
        "type": "intro",
        "title": "CUSTOM VIDEO",
        "subtitle": "My Analysis",
        "duration": 4.0,
    },
    {
        "type": "stats",
        "title": "KEY STATISTICS",
        "data": [
            {"label": "Hits", "value": 12},
            {"label": "Home Runs", "value": 3},
            {"label": "RBI", "value": 8},
        ],
        "duration": 5.0,
    },
    {
        "type": "chart",
        "chart_type": "bar",
        "title": "Runs by Inning",
        "data": {
            "labels": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "values": [0, 2, 0, 1, 0, 3, 0, 1, 0],
        },
        "duration": 6.0,
    },
]

video_path = generator.create_video(scenes, output_name="custom_video")
```

## Troubleshooting

### "OpenAI API key not configured"

Make sure your `.env` file has the key and it's loaded:

```bash
source mlb-env/bin/activate  # Reactivate environment
python -c "from config.settings import settings; print(settings.openai_api_key[:10])"
```

### MoviePy errors

Install FFmpeg:

```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### "No games found"

Check the date format (YYYY-MM-DD) and that there were actually games:

```bash
python scripts/fetch_games.py --date 2024-12-25  # No games on Christmas!
python scripts/fetch_games.py --date 2024-07-04  # Games on July 4th
```

## Next Steps

1. **Customize prompts**: Edit `src/content/prompts.py` for different script styles
2. **Add templates**: Create new templates in `src/video/templates.py`
3. **Automate**: Set up n8n workflows in `deployment/n8n/`
4. **Deploy**: Use Docker configs in `deployment/docker/`
5. **Scale**: Deploy to AWS Lambda with configs in `deployment/lambda/`

## Getting Help

- Check the [README.md](../README.md) for detailed documentation
- Run tests: `python scripts/test_pipeline.py --verbose`
- View logs: `cat logs/pipeline.log`
- Check costs: `cat logs/costs.json`
