"""
Tests for prediction model.

Run: pytest tests/test_predictor.py -v
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path

from src.models.predictor import PlayerPredictor, BattingPredictor, EnsemblePredictor
from src.models.trainer import ModelTrainer, EarlyStopping


class TestPlayerPredictor:
    """Tests for PlayerPredictor class."""

    def test_init(self):
        """Test model initialization."""
        model = PlayerPredictor(input_features=10, hidden_dim=32, output_dim=4)

        assert model.input_features == 10
        assert model.hidden_dim == 32
        assert model.output_dim == 4

    def test_forward_pass(self):
        """Test forward pass dimensions."""
        model = PlayerPredictor(input_features=10, hidden_dim=32, output_dim=4)

        # Batch of 5 samples
        x = torch.randn(5, 10)
        y = model(x)

        assert y.shape == (5, 4)

    def test_single_sample(self):
        """Test with single sample."""
        model = PlayerPredictor(input_features=10, hidden_dim=32, output_dim=4)
        model.eval()

        x = torch.randn(1, 10)
        y = model(x)

        assert y.shape == (1, 4)

    def test_predict_numpy(self):
        """Test prediction with numpy input."""
        model = PlayerPredictor(input_features=10, hidden_dim=32, output_dim=4)

        x = np.random.randn(5, 10).astype(np.float32)
        y = model.predict(x, return_numpy=True)

        assert isinstance(y, np.ndarray)
        assert y.shape == (5, 4)

    def test_predict_single_numpy(self):
        """Test prediction with single numpy array."""
        model = PlayerPredictor(input_features=10, hidden_dim=32, output_dim=4)

        x = np.random.randn(10).astype(np.float32)
        y = model.predict(x, return_numpy=True)

        assert y.shape == (1, 4)

    def test_save_and_load(self):
        """Test model save and load."""
        model = PlayerPredictor(input_features=10, hidden_dim=32, output_dim=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pth"

            # Save
            model.save(path)
            assert path.exists()

            # Load
            loaded = PlayerPredictor.load(path)
            assert loaded.input_features == 10
            assert loaded.hidden_dim == 32
            assert loaded.output_dim == 4

            # Verify weights are the same
            x = torch.randn(1, 10)
            model.eval()
            loaded.eval()
            y1 = model(x)
            y2 = loaded(x)

            assert torch.allclose(y1, y2)


class TestBattingPredictor:
    """Tests for BattingPredictor class."""

    def test_feature_names(self):
        """Test that feature names are defined."""
        assert len(BattingPredictor.FEATURE_NAMES) > 0
        assert len(BattingPredictor.OUTPUT_NAMES) == 4

    def test_init(self):
        """Test initialization with correct dimensions."""
        model = BattingPredictor()

        assert model.input_features == len(BattingPredictor.FEATURE_NAMES)
        assert model.output_dim == len(BattingPredictor.OUTPUT_NAMES)

    def test_interpret_predictions(self):
        """Test prediction interpretation."""
        model = BattingPredictor()

        predictions = np.array([[1.5, 0.3, 2.1, 0.8]])
        results = model.interpret_predictions(predictions)

        assert len(results) == 1
        assert "hits" in results[0]
        assert "home_runs" in results[0]
        assert results[0]["hits"] >= 0  # Non-negative


class TestEnsemblePredictor:
    """Tests for EnsemblePredictor class."""

    def test_equal_weights(self):
        """Test ensemble with equal weights."""
        model1 = PlayerPredictor(input_features=10, hidden_dim=16, output_dim=4)
        model2 = PlayerPredictor(input_features=10, hidden_dim=16, output_dim=4)

        ensemble = EnsemblePredictor([model1, model2])

        x = np.random.randn(5, 10).astype(np.float32)
        y = ensemble.predict(x)

        assert y.shape == (5, 4)

    def test_custom_weights(self):
        """Test ensemble with custom weights."""
        model1 = PlayerPredictor(input_features=10, hidden_dim=16, output_dim=4)
        model2 = PlayerPredictor(input_features=10, hidden_dim=16, output_dim=4)

        ensemble = EnsemblePredictor([model1, model2], weights=[0.7, 0.3])

        x = np.random.randn(5, 10).astype(np.float32)
        y = ensemble.predict(x)

        assert y.shape == (5, 4)

    def test_invalid_weights(self):
        """Test that invalid weights raise error."""
        model1 = PlayerPredictor(input_features=10, hidden_dim=16, output_dim=4)
        model2 = PlayerPredictor(input_features=10, hidden_dim=16, output_dim=4)

        with pytest.raises(ValueError):
            EnsemblePredictor([model1, model2], weights=[0.5, 0.3])  # Doesn't sum to 1


class TestModelTrainer:
    """Tests for ModelTrainer class."""

    @pytest.fixture
    def trainer(self):
        """Create trainer with small model."""
        model = PlayerPredictor(input_features=10, hidden_dim=16, output_dim=4)
        return ModelTrainer(model, learning_rate=0.01)

    def test_create_data_loader(self, trainer):
        """Test data loader creation."""
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randn(100, 4).astype(np.float32)

        loader = trainer.create_data_loader(X, y, batch_size=32)

        assert len(loader) == 4  # 100 / 32 = 3.125, rounds up

    def test_train_epoch(self, trainer):
        """Test single training epoch."""
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randn(100, 4).astype(np.float32)

        loader = trainer.create_data_loader(X, y, batch_size=32)
        loss = trainer.train_epoch(loader)

        assert loss > 0
        assert not np.isnan(loss)

    def test_validate(self, trainer):
        """Test validation."""
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randn(50, 4).astype(np.float32)

        loader = trainer.create_data_loader(X, y, batch_size=32, shuffle=False)
        loss = trainer.validate(loader)

        assert loss > 0

    def test_full_training(self, trainer):
        """Test full training loop."""
        X_train = np.random.randn(100, 10).astype(np.float32)
        y_train = np.random.randn(100, 4).astype(np.float32)
        X_val = np.random.randn(30, 10).astype(np.float32)
        y_val = np.random.randn(30, 4).astype(np.float32)

        train_loader = trainer.create_data_loader(X_train, y_train)
        val_loader = trainer.create_data_loader(X_val, y_val, shuffle=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            history = trainer.train(
                train_loader,
                val_loader,
                epochs=5,
                checkpoint_dir=Path(tmpdir),
                verbose=False,
            )

        assert "train_loss" in history
        assert len(history["train_loss"]) == 5

    def test_evaluate(self, trainer):
        """Test model evaluation."""
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randn(50, 4).astype(np.float32)

        loader = trainer.create_data_loader(X, y, shuffle=False)
        metrics = trainer.evaluate(loader)

        assert "mse" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics


class TestEarlyStopping:
    """Tests for EarlyStopping class."""

    def test_no_improvement(self):
        """Test early stopping triggers after patience."""
        early_stopping = EarlyStopping(patience=3)

        # Simulate no improvement
        for loss in [1.0, 1.0, 1.0, 1.0]:
            should_stop = early_stopping(loss)

        assert should_stop is True

    def test_with_improvement(self):
        """Test early stopping doesn't trigger with improvement."""
        early_stopping = EarlyStopping(patience=3)

        # Simulate improvement
        losses = [1.0, 0.9, 0.8, 0.7]
        for loss in losses:
            should_stop = early_stopping(loss)

        assert should_stop is False

    def test_min_delta(self):
        """Test minimum improvement threshold."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.1)

        # Small improvements below threshold
        early_stopping(1.0)
        early_stopping(0.99)  # Improvement < min_delta
        should_stop = early_stopping(0.98)

        assert should_stop is True
