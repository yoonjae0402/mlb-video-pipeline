#!/usr/bin/env python3
"""
MLB Video Pipeline - Train Model Script

CLI tool to train the player performance prediction model.

Usage:
    python scripts/train_model.py --data training_data.csv
    python scripts/train_model.py --epochs 100 --learning-rate 0.001
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.models.predictor import PlayerPredictor, BattingPredictor
from src.models.trainer import ModelTrainer
from src.utils.logger import setup_logging, get_logger


console = Console()
logger = get_logger(__name__)


@click.command()
@click.option(
    "--data", "-d",
    type=click.Path(exists=True),
    help="Path to training data CSV"
)
@click.option(
    "--epochs", "-e",
    type=int,
    default=50,
    help="Number of training epochs"
)
@click.option(
    "--learning-rate", "-lr",
    type=float,
    default=0.001,
    help="Learning rate"
)
@click.option(
    "--batch-size", "-b",
    type=int,
    default=32,
    help="Batch size"
)
@click.option(
    "--hidden-dim",
    type=int,
    default=64,
    help="Hidden layer dimension"
)
@click.option(
    "--val-split",
    type=float,
    default=0.2,
    help="Validation split ratio"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output path for trained model"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Verbose output"
)
def main(
    data: str | None,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    hidden_dim: int,
    val_split: float,
    output: str | None,
    verbose: bool,
):
    """Train the player performance prediction model."""

    setup_logging(level="DEBUG" if verbose else "INFO")

    console.print("[bold blue]MLB Video Pipeline - Model Training[/bold blue]")
    console.print()

    # Load or generate training data
    if data:
        console.print(f"Loading training data from {data}...")
        df = pd.read_csv(data)
    else:
        console.print("[yellow]No data provided, generating synthetic data for demo...[/yellow]")
        df = _generate_demo_data()

    console.print(f"Training data shape: {df.shape}")

    # Prepare features and targets
    feature_cols = [col for col in df.columns if col.startswith("feature_") or col.startswith("rolling_")]
    target_cols = [col for col in df.columns if col.startswith("next_") or col.startswith("target_")]

    if not feature_cols:
        # Use demo columns
        feature_cols = [f"feature_{i}" for i in range(10)]
        for col in feature_cols:
            if col not in df.columns:
                df[col] = np.random.randn(len(df))

    if not target_cols:
        target_cols = ["target_0"]
        df["target_0"] = np.random.randn(len(df))

    console.print(f"Features: {len(feature_cols)}, Targets: {len(target_cols)}")

    # Split data
    from sklearn.model_selection import train_test_split

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=42
    )

    console.print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Create model
    model = PlayerPredictor(
        input_features=len(feature_cols),
        hidden_dim=hidden_dim,
        output_dim=len(target_cols),
    )

    # Create trainer
    trainer = ModelTrainer(
        model=model,
        learning_rate=learning_rate,
    )

    # Create data loaders
    train_loader = trainer.create_data_loader(X_train, y_train, batch_size=batch_size)
    val_loader = trainer.create_data_loader(X_val, y_val, batch_size=batch_size, shuffle=False)

    # Train
    console.print()
    console.print("[bold]Starting training...[/bold]")

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        verbose=True,
    )

    # Evaluate
    console.print()
    console.print("[bold]Evaluating model...[/bold]")

    metrics = trainer.evaluate(val_loader)

    console.print()
    console.print("[bold green]Training Complete![/bold green]")
    console.print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
    console.print(f"  Final Val Loss: {history['val_loss'][-1]:.4f}")
    console.print(f"  R² Score: {metrics['r2']:.4f}")
    console.print(f"  MAE: {metrics['mae']:.4f}")

    # Save model
    output_path = output or settings.models_dir / "player_predictor.pth"
    model.save(output_path)
    console.print(f"\n[green]Model saved to: {output_path}[/green]")


def _generate_demo_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic training data for demonstration."""
    np.random.seed(42)

    data = {}

    # Generate features
    for i in range(10):
        data[f"feature_{i}"] = np.random.randn(n_samples)

    # Generate target (simple linear combination + noise)
    target = sum(data[f"feature_{i}"] * (i + 1) for i in range(10))
    target = target + np.random.randn(n_samples) * 0.5
    data["target_0"] = target

    return pd.DataFrame(data)


if __name__ == "__main__":
    main()
