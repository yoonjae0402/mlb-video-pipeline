"""
MLB Video Pipeline - Models Package

PyTorch-based prediction models for player performance:
- classifier: LSTM classifier (3-class: below/average/above)
- predictor: Legacy regression predictor (kept for compatibility)
- trainer: Training loop and evaluation

Usage:
    from src.models import PlayerPredictor, ModelTrainer

    # Create classifier model
    from src.models.classifier import PlayerPredictor

    model = PlayerPredictor(input_features=11, hidden_dim=64)

    # Train model
    trainer = ModelTrainer(model)
    trainer.train(train_loader, val_loader, epochs=50)

    # Make predictions
    predictions = model.predict(features)
"""

# Import from both old and new locations for compatibility
from src.models.predictor import PlayerPredictor
from src.models.trainer import ModelTrainer

# New classifier imports
try:
    from src.models.classifier import (
        PlayerPredictor as PlayerClassifier,
        BattingPredictor,
        PitchingPredictor,
        EnsemblePredictor,
    )
except ImportError:
    PlayerClassifier = PlayerPredictor
    BattingPredictor = None
    PitchingPredictor = None
    EnsemblePredictor = None

__all__ = [
    "PlayerPredictor",
    "PlayerClassifier",
    "BattingPredictor",
    "PitchingPredictor",
    "EnsemblePredictor",
    "ModelTrainer",
]
