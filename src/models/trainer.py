"""
MLB Video Pipeline - Model Trainer

Training loop, validation, and evaluation for PyTorch models.
Includes early stopping, learning rate scheduling, and metrics tracking.

Usage:
    from src.models.trainer import ModelTrainer
    from src.models.predictor import PlayerPredictor

    model = PlayerPredictor()
    trainer = ModelTrainer(model, learning_rate=0.001)

    # Train the model
    history = trainer.train(train_loader, val_loader, epochs=50)

    # Evaluate
    metrics = trainer.evaluate(test_loader)
"""

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

from config.settings import settings
from src.utils.logger import get_logger


logger = get_logger(__name__)


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors validation loss and stops training when it stops improving.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum improvement to reset patience
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class ModelTrainer:
    """
    Trainer for PyTorch prediction models.

    Handles training loop, validation, learning rate scheduling,
    checkpointing, and metrics tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        device: str | None = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            device: Device to train on (cpu/cuda/mps)
        """
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Set device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # Move model to device
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Default loss function
        self.criterion = nn.MSELoss()

        # Training history
        self.history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        logger.info(f"Trainer initialized on {self.device}")

    def create_data_loader(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Create a DataLoader from numpy arrays.

        Args:
            features: Input features
            targets: Target values
            batch_size: Batch size
            shuffle: Whether to shuffle data

        Returns:
            PyTorch DataLoader
        """
        features_tensor = torch.FloatTensor(features)
        targets_tensor = torch.FloatTensor(targets)
        dataset = TensorDataset(features_tensor, targets_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for features, targets in train_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.criterion(predictions, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(features)
                loss = self.criterion(predictions, targets)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 50,
        patience: int = 10,
        checkpoint_dir: Path | None = None,
        verbose: bool = True,
    ) -> dict[str, list]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            patience: Early stopping patience
            checkpoint_dir: Directory for checkpoints
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {epochs} epochs")

        # Setup
        early_stopping = EarlyStopping(patience=patience)
        scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        best_val_loss = float("inf")
        checkpoint_dir = checkpoint_dir or settings.models_dir

        # Training loop
        progress = tqdm(range(epochs), desc="Training", disable=not verbose)

        for epoch in progress:
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)
                scheduler.step(val_loss)

                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(checkpoint_dir / "best_model.pth")

                # Early stopping
                if early_stopping(val_loss):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                progress.set_postfix({
                    "train_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                })
            else:
                progress.set_postfix({"train_loss": f"{train_loss:.4f}"})

            # Track learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["learning_rate"].append(current_lr)

        # Save final model
        self._save_checkpoint(checkpoint_dir / "final_model.pth")
        logger.info("Training complete")

        return self.history

    def _save_checkpoint(self, path: Path) -> None:
        """Save training checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
        logger.info(f"Loaded checkpoint from {path}")

    def evaluate(self, test_loader: DataLoader) -> dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(self.device)
                predictions = self.model(features)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.numpy())

        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)

        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)

        # R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        metrics = {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        }

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            features: Input features

        Returns:
            Model predictions
        """
        self.model.eval()
        features_tensor = torch.FloatTensor(features).to(self.device)

        if features_tensor.dim() == 1:
            features_tensor = features_tensor.unsqueeze(0)

        with torch.no_grad():
            predictions = self.model(features_tensor)

        return predictions.cpu().numpy()
