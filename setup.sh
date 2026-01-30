#!/bin/bash
# MLB Video Pipeline - Project Setup Script
# Run: chmod +x setup.sh && ./setup.sh

set -e  # Exit on error

echo "🏟️  Setting up MLB Video Pipeline..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create directory structure
echo -e "${YELLOW}Creating directories...${NC}"

directories=(
    "config"
    "data/raw"
    "data/processed"
    "data/cache"
    "models/training_data"
    "src/data"
    "src/models"
    "src/content"
    "src/audio"
    "src/video/assets"
    "src/upload"
    "src/utils"
    "scripts"
    "outputs/scripts"
    "outputs/audio"
    "outputs/videos"
    "outputs/thumbnails"
    "logs"
    "tests"
    "dashboard"
    "deployment/lambda"
    "deployment/n8n"
    "deployment/docker"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    echo "  Created: $dir"
done

# Create __init__.py files
echo -e "${YELLOW}Creating __init__.py files...${NC}"

init_dirs=(
    "config"
    "src"
    "src/data"
    "src/models"
    "src/content"
    "src/audio"
    "src/video"
    "src/upload"
    "src/utils"
    "tests"
)

for dir in "${init_dirs[@]}"; do
    touch "$dir/__init__.py"
    echo "  Created: $dir/__init__.py"
done

# Create .gitkeep files for empty directories
echo -e "${YELLOW}Creating .gitkeep files...${NC}"

gitkeep_dirs=(
    "data/raw"
    "data/processed"
    "data/cache"
    "models/training_data"
    "src/video/assets"
    "outputs/scripts"
    "outputs/audio"
    "outputs/videos"
    "outputs/thumbnails"
    "logs"
    "deployment/lambda"
    "deployment/n8n"
    "deployment/docker"
)

for dir in "${gitkeep_dirs[@]}"; do
    touch "$dir/.gitkeep"
done

# Setup virtual environment
echo -e "${YELLOW}Setting up virtual environment...${NC}"

if [ ! -d "mlb-env" ]; then
    python3 -m venv mlb-env
    echo "  Created virtual environment: mlb-env"
else
    echo "  Virtual environment already exists"
fi

# Activate and install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
source mlb-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Setup environment file
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${YELLOW}Created .env from .env.example - please add your API keys${NC}"
fi

# Initialize logs
touch logs/pipeline.log
echo '{"daily_calls": {}, "total_cost": 0}' > logs/api_usage.json
echo '{"openai": 0, "elevenlabs": 0, "total": 0}' > logs/costs.json

echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys"
echo "  2. Activate environment: source mlb-env/bin/activate"
echo "  3. Test setup: python scripts/test_pipeline.py"
echo ""
