"""Model checkpointing utilities for RL training.

Provides functionality to save and load model checkpoints,
including tracking the best model based on a metric.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class ModelCheckpointer:
    """Handles saving and loading of model checkpoints.

    Tracks the best model based on a metric (e.g., evaluation reward)
    and provides utilities for saving checkpoints at various points
    during training.
    """

    def __init__(self, save_dir: str | Path):
        """Initialize the checkpointer.

        Args:
            save_dir: Directory to save model checkpoints.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._best_metric: float = -np.inf

    @property
    def best_metric(self) -> float:
        """Return the best metric value seen so far."""
        return self._best_metric

    def save(
        self,
        models: dict[str, nn.Module],
        suffix: str = "",
    ) -> None:
        """Save model checkpoints.

        Args:
            models: Dictionary mapping model names to PyTorch modules.
            suffix: Optional suffix to append to filenames (e.g., "_100" for iteration 100).
        """
        for name, model in models.items():
            filename = f"{name}{suffix}.pt"
            path = self.save_dir / filename
            torch.save(model, path)
            print(f"Saved {name} at {path}")

    def save_if_best(
        self,
        models: dict[str, nn.Module],
        metric: float,
        step: int,
    ) -> bool:
        """Save models if the metric is the best seen so far.

        Always saves with the iteration suffix. Additionally saves
        without suffix (as "best" checkpoint) if metric improves.

        Args:
            models: Dictionary mapping model names to PyTorch modules.
            metric: Current metric value to compare against best.
            step: Current training step/iteration.

        Returns:
            True if this was a new best, False otherwise.
        """
        # Always save with iteration suffix
        self.save(models, suffix=f"_{step}")

        # Check if this is a new best
        is_new_best = metric > self._best_metric
        if is_new_best:
            self._best_metric = metric
            # Save without suffix as "best" checkpoint
            self.save(models)

        return is_new_best

    def get_checkpoint_path(self, model_name: str, suffix: str = "") -> Path:
        """Get the path for a checkpoint file.

        Args:
            model_name: Name of the model (e.g., "actor", "critic").
            suffix: Optional suffix (e.g., "_100").

        Returns:
            Full path to the checkpoint file.
        """
        return self.save_dir / f"{model_name}{suffix}.pt"

    def load(
        self,
        model_name: str,
        suffix: str = "",
        map_location: str | None = None,
    ) -> nn.Module:
        """Load a model checkpoint.

        Args:
            model_name: Name of the model to load.
            suffix: Optional suffix for the checkpoint.
            map_location: Device to map the loaded model to.

        Returns:
            Loaded PyTorch module.

        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist.
        """
        path = self.get_checkpoint_path(model_name, suffix)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch.load(path, map_location=map_location, weights_only=False)

    def list_checkpoints(self, model_name: str = "actor") -> list:
        """List all available checkpoints for a model.

        Args:
            model_name: Name of the model to list checkpoints for.

        Returns:
            List of checkpoint suffixes (e.g., ["", "_100", "_200"]).
        """
        pattern = f"{model_name}*.pt"
        checkpoints = list(self.save_dir.glob(pattern))
        suffixes = []
        for cp in checkpoints:
            stem = cp.stem
            if stem == model_name:
                suffixes.append("")
            else:
                suffixes.append(stem[len(model_name) :])
        return sorted(suffixes)
