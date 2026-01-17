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

    Uses a combination scheme that avoids collisions while staying within
    numpy's seed range (0 to 2^32-1). The scheme uses modular arithmetic
    to ensure valid seed values.

    Args:
        master_seed: The master seed from CLI
        worker_id: Worker index (0-based)
        offset: Phase offset (0=training workers, 1=normalization, etc.)

    Returns:
        Deterministic seed for this worker (0 to 2^32-1)
    """
    # Use a large prime multiplier to spread seeds and avoid collisions
    # Keep result within numpy's valid seed range (2^32 - 1)
    MAX_SEED = 2**32 - 1
    combined = master_seed * 1_000_003 + offset * 10_007 + worker_id
    return combined % MAX_SEED
