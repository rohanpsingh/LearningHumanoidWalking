"""Deterministic seeding utilities for reproducible training."""

import os
import random
import numpy as np
import torch


def set_global_seeds(seed: int, cuda_deterministic: bool = True) -> None:
    """Set global seeds for all random number generators.

    Args:
        seed: Master seed value
        cuda_deterministic: If True, enable CUDA deterministic mode
                           (may reduce performance)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if cuda_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Set PYTHONHASHSEED for dict ordering determinism
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_worker_seed(master_seed: int, worker_id: int, offset: int = 0) -> int:
    """Derive deterministic worker-specific seed.

    Uses multiplication to avoid collisions between different worker_id/offset
    combinations. With this scheme:
    - offset separates phases (0=training, 1=normalization, etc.)
    - worker_id separates workers within a phase
    - Supports up to 10000 workers per phase and 1000 phases

    Args:
        master_seed: The master seed from CLI
        worker_id: Worker index (0-based)
        offset: Phase offset (0=training workers, 1=normalization, etc.)

    Returns:
        Deterministic seed for this worker
    """
    return master_seed * 1_000_000 + offset * 10_000 + worker_id
