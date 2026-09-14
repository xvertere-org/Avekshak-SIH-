"""
Train/validation/test splitting utilities.

Key principle: splits are performed at the UNIT level (engine, bearing, battery),
never by randomly splitting windows from the same unit across partitions.
"""

import numpy as np
from typing import List, Tuple, Optional, Set


def split_by_id(
    ids: List[str],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split a list of unique IDs into train/val/test partitions.

    This ensures no ID appears in more than one partition,
    preventing data leakage for bearing/engine/battery-level splits.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    rng = np.random.RandomState(seed)
    unique_ids = sorted(set(ids))
    rng.shuffle(unique_ids)

    n = len(unique_ids)
    n_train = max(1, int(n * train_ratio))
    n_val = max(0, int(n * val_ratio))
    # Test gets the remainder
    train_ids = unique_ids[:n_train]
    val_ids = unique_ids[n_train:n_train + n_val]
    test_ids = unique_ids[n_train + n_val:]

    return train_ids, val_ids, test_ids


def verify_no_leakage(
    train_ids: Set[str], val_ids: Set[str], test_ids: Set[str]
) -> bool:
    """Verify that no ID appears in more than one partition."""
    return (
        len(train_ids & val_ids) == 0
        and len(train_ids & test_ids) == 0
        and len(val_ids & test_ids) == 0
    )
