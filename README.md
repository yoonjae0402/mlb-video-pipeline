# MLB Automated Video Pipeline ⚾

Automated system for generating baseball analysis videos from MLB game data. No copyrighted footage - pure statistics, AI narration, and data visualization.

## Features

- **Data Collection**: Fetch game data, player stats, and standings from MLB Stats API
- **ML Predictions**: PyTorch models for player performance predictions
- **AI Scripts**: GPT-4 powered script generation for engaging narration
- **Natural TTS**: ElevenLabs text-to-speech for professional voiceover
- **Video Generation**: Automated video creation with stats, charts, and graphics
- **YouTube Upload**: Direct upload to YouTube with metadata
- **Cost Tracking**: Monitor API usage and spending
- **Dashboard**: Streamlit monitoring interface

## Quick Start

### 1. Clone and Setup

```bash
git clone <repo-url>
cd mlb-video-pipeline

# Create virtual environment
python3 -m venv mlb-env
source mlb-env/bin/activate  # On Windows: mlb-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required API keys:
- **OpenAI**: Get from [platform.openai.com](https://platform.openai.com/api-keys)
- **ElevenLabs**: Get from [elevenlabs.io](https://elevenlabs.io/)
- **YouTube** (optional): From [Google Cloud Console](https://console.cloud.google.com/)

### 3. Run Tests

```bash
python scripts/test_pipeline.py
```

### 4. Generate Your First Video

```bash
# Fetch yesterday's games
python scripts/fetch_games.py

# Generate a video (dry run first)
python scripts/generate_video.py --dry-run

# Generate for real
python scripts/generate_video.py --date 2024-07-04
```

## Project Structure

```
mlb-video-pipeline/
├── config/                  # Configuration
│   ├── settings.py          # API keys, paths, constants
│   └── league_config.py     # MLB teams, stats categories
├── data/                    # Data storage
│   ├── raw/                 # Raw MLB API responses
│   ├── processed/           # Cleaned data
│   └── cache/               # API response cache
├── models/                  # ML models
│   ├── player_predictor.pth # Trained model
│   └── training_data/       # Historical data
├── src/                     # Source code
│   ├── data/                # Data fetching & processing
│   ├── models/              # PyTorch model & trainer
│   ├── content/             # GPT script generation
│   ├── audio/               # ElevenLabs TTS
│   ├── video/               # Video generation
│   ├── upload/              # YouTube upload
│   └── utils/               # Logging, validation
├── scripts/                 # CLI tools
│   ├── fetch_games.py       # Fetch MLB data
│   ├── train_model.py       # Train prediction model
│   ├── generate_video.py    # Create videos
│   └── test_pipeline.py     # Run tests
├── outputs/                 # Generated content
│   ├── scripts/             # Text scripts
│   ├── audio/               # MP3 narration
│   ├── videos/              # Final videos
│   └── thumbnails/          # Video thumbnails
├── tests/                   # Unit tests
├── dashboard/               # Streamlit dashboard
└── deployment/              # AWS, n8n, Docker configs
```

## Usage Examples

### Fetch Game Data

```python
from src.data import MLBDataFetcher

fetcher = MLBDataFetcher()
games = fetcher.get_games_for_date("2024-07-04")

for game in games:
    print(f"{game['away_team_name']} @ {game['home_team_name']}: {game['status']}")
```

### Generate Script

```python
from src.content import ScriptGenerator

generator = ScriptGenerator()
script = generator.generate_game_recap({
    "game_id": 12345,
    "game_date": "2024-07-04",
    "home_team_id": 147,  # Yankees
    "away_team_id": 111,  # Red Sox
    "home_score": 5,
    "away_score": 3,
    "key_stats": {"winning_pitcher": "Gerrit Cole"},
    "notable_performances": "Aaron Judge: 2 HR, 4 RBI",
}, duration=60)
```

### Create Video

```python
from src.video import VideoGenerator

generator = VideoGenerator(template="modern_dark")

video_path = generator.create_video(
    scenes=[
        {"type": "intro", "title": "GAME RECAP", "duration": 3},
        {"type": "stats", "data": stats_list, "duration": 5},
    ],
    audio_path=narration_path,
    output_name="yankees_recap",
)
```

### Train Prediction Model

```python
from src.models import PlayerPredictor, ModelTrainer

model = PlayerPredictor(input_features=15, hidden_dim=64)
trainer = ModelTrainer(model, learning_rate=0.001)

history = trainer.train(train_loader, val_loader, epochs=50)
model.save("models/player_predictor.pth")
```

## CLI Commands

```bash
# Fetch games
python scripts/fetch_games.py --date 2024-07-04
python scripts/fetch_games.py --team NYY --date 2024-07-04
python scripts/fetch_games.py --start 2024-07-01 --end 2024-07-07

# Train model
python scripts/train_model.py --epochs 100 --learning-rate 0.001

# Generate video
python scripts/generate_video.py --date 2024-07-04
python scripts/generate_video.py --game-id 12345 --template classic_baseball
python scripts/generate_video.py --type player_spotlight --dry-run

# Run tests
python scripts/test_pipeline.py
python scripts/test_pipeline.py --component data
pytest tests/ -v
```

## Dashboard

Launch the monitoring dashboard:

```bash
streamlit run dashboard/streamlit_app.py
```

Features:
- Pipeline overview and status
- Cost tracking and budget monitoring
- Video management
- Data file browser

## Video Templates

Available templates:
- `modern_dark`: Sleek dark theme with vibrant accents
- `classic_baseball`: Traditional baseball aesthetic
- `electric_neon`: High-energy style for highlights
- `clean_minimal`: Simple, professional look
- `team_branded`: Customizable with team colors

## Cost Management

The pipeline tracks all API costs:

```python
from src.utils.logger import CostTracker

tracker = CostTracker()
print(f"Today's spend: ${tracker.get_daily_total():.2f}")
print(f"Under budget: {tracker.check_budget(10.0)}")
```

Default limits:
- Daily spend limit: $10
- OpenAI max tokens: 500 per request
- ElevenLabs monthly chars: 100,000

## Deployment

### AWS Lambda

```bash
cd deployment/lambda
# Deploy using AWS SAM or Serverless Framework
```

### n8n Automation

Import workflows from `deployment/n8n/` into your n8n instance.

### Docker

```bash
cd deployment/docker
docker build -t mlb-pipeline .
docker run -v ./data:/app/data mlb-pipeline
```

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Integration tests (requires network)
pytest tests/ -v -m integration
```

### Code Style

```bash
# Format code
black src/ scripts/ tests/

# Sort imports
isort src/ scripts/ tests/

# Type checking
mypy src/
```

## Troubleshooting

### API Key Issues

```bash
# Verify keys are loaded
python -c "from config.settings import settings; print(settings.validate_api_keys())"
```

### MoviePy/FFmpeg Issues

```bash
# Install FFmpeg
brew install ffmpeg  # macOS
apt install ffmpeg   # Ubuntu
```

### Cache Issues

```bash
# Clear API cache
python -c "from src.data import MLBDataFetcher; MLBDataFetcher().clear_cache()"
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

---

Built with Python, PyTorch, GPT-4, ElevenLabs, and MoviePy.
