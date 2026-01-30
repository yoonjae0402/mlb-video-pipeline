"""
MLB Video Pipeline - PyTorch LSTM Classifier

PyTorch LSTM classifier for player performance prediction.
3-class classification: below_average / average / above_average

Architecture:
- Input: (batch, 10 timesteps, 11 features)
- LSTM: 2 layers, 64 hidden units
- FC: 64 → 32 → 3 classes
- Output: Softmax probabilities

The 11 input features per timestep:
1-3: Batting avg, OBP, SLG (rolling 10-game)
4-6: HR, RBI, Hits (per game)
7-9: AB, BB, K (per game)
10: is_home (0/1)
11: season_break (0/1, indicates cross-season boundary)

Usage:
    from src.models.classifier import PlayerClassifier

    # Initialize model
    model = PlayerClassifier()

    # Forward pass
    logits = model(input_tensor)  # (batch, 3)

    # Get class predictions
    predictions = model.predict(input_tensor)  # (batch,)

    # Load trained model
    model = PlayerClassifier.load("models/player_classifier.pth")
"""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import numpy as np

from config.settings import settings
from src.utils.logger import get_logger


logger = get_logger(__name__)


class PlayerPredictor(nn.Module):
    """
    Neural network for predicting player performance.

    Architecture:
    - Input layer: Player features (rolling stats, matchup data)
    - Hidden layers: Fully connected with ReLU and dropout
    - Output layer: Predicted statistics

    Attributes:
        input_features: Number of input features
        hidden_dim: Hidden layer dimension
        output_dim: Number of output predictions
    """

    def __init__(
        self,
        input_features: int = 15,
        hidden_dim: int = 64,
        output_dim: int = 4,
        dropout: float = 0.3,
    ):
        """
        Initialize the predictor model.

        Args:
            input_features: Number of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Number of output predictions
            dropout: Dropout probability
        """
        super().__init__()

        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Define network architecture
        self.network = nn.Sequential(
            # First hidden layer
            nn.Linear(input_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Second hidden layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Third hidden layer
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),

            # Output layer
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # Initialize weights
        self._initialize_weights()

        logger.info(
            f"PlayerPredictor initialized: "
            f"{input_features} -> {hidden_dim} -> {output_dim}"
        )

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_features)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)

    def predict(
        self,
        features: np.ndarray | torch.Tensor,
        return_numpy: bool = True
    ) -> np.ndarray | torch.Tensor:
        """
        Make predictions for given features.

        Args:
            features: Input features
            return_numpy: Whether to return numpy array

        Returns:
            Predicted values
        """
        self.eval()

        # Convert to tensor if needed
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)

        # Ensure batch dimension
        if features.dim() == 1:
            features = features.unsqueeze(0)

        with torch.no_grad():
            predictions = self.forward(features)

        if return_numpy:
            return predictions.numpy()
        return predictions

    def save(self, path: str | Path) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_config": {
                "input_features": self.input_features,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "PlayerPredictor":
        """
        Load model from disk.

        Args:
            path: Path to saved model

        Returns:
            Loaded model instance
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Create model with saved config
        config = checkpoint["model_config"]
        model = cls(
            input_features=config["input_features"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["output_dim"],
        )

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        logger.info(f"Model loaded from {path}")
        return model


class BattingPredictor(PlayerPredictor):
    """
    Specialized predictor for batting statistics.

    Predicts: hits, home runs, RBIs, walks
    """

    # Default feature names for batting prediction
    FEATURE_NAMES = [
        "rolling_avg_10",
        "rolling_obp_10",
        "rolling_slg_10",
        "rolling_hr_10",
        "vs_pitcher_avg",
        "home_away",
        "day_night",
        "park_factor",
        "recent_form",
        "days_rest",
    ]

    OUTPUT_NAMES = ["hits", "home_runs", "rbi", "walks"]

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__(
            input_features=len(self.FEATURE_NAMES),
            hidden_dim=hidden_dim,
            output_dim=len(self.OUTPUT_NAMES),
            dropout=dropout,
        )

    def interpret_predictions(
        self,
        predictions: np.ndarray
    ) -> list[dict[str, float]]:
        """
        Convert raw predictions to labeled dictionary.

        Args:
            predictions: Raw model output

        Returns:
            List of dictionaries with labeled predictions
        """
        if predictions.ndim == 1:
            predictions = predictions.reshape(1, -1)

        results = []
        for pred in predictions:
            results.append({
                name: float(max(0, value))  # Ensure non-negative
                for name, value in zip(self.OUTPUT_NAMES, pred)
            })
        return results


class PitchingPredictor(PlayerPredictor):
    """
    Specialized predictor for pitching statistics.

    Predicts: innings pitched, strikeouts, walks, earned runs
    """

    FEATURE_NAMES = [
        "rolling_era_5",
        "rolling_whip_5",
        "rolling_k9_5",
        "vs_team_era",
        "home_away",
        "day_night",
        "park_factor",
        "days_rest",
        "pitch_count_last",
        "season_workload",
    ]

    OUTPUT_NAMES = ["innings_pitched", "strikeouts", "walks", "earned_runs"]

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__(
            input_features=len(self.FEATURE_NAMES),
            hidden_dim=hidden_dim,
            output_dim=len(self.OUTPUT_NAMES),
            dropout=dropout,
        )


class EnsemblePredictor:
    """
    Ensemble of multiple models for robust predictions.

    Combines predictions from multiple models using averaging
    or weighted voting.
    """

    def __init__(self, models: list[PlayerPredictor], weights: list[float] | None = None):
        """
        Initialize ensemble predictor.

        Args:
            models: List of predictor models
            weights: Optional weights for each model (must sum to 1)
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

        if len(self.weights) != len(models):
            raise ValueError("Number of weights must match number of models")

        if abs(sum(self.weights) - 1.0) > 0.001:
            raise ValueError("Weights must sum to 1")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make ensemble prediction.

        Args:
            features: Input features

        Returns:
            Weighted average of model predictions
        """
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(features, return_numpy=True)
            predictions.append(pred * weight)

        return np.sum(predictions, axis=0)
