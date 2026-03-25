"""
Shared helpers for ReID dataset loaders.

These are intentionally tiny and dependency-light so individual dataset loader
files stay simple.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Sequence, Tuple, TypeVar

import numpy as np

T = TypeVar("T")


def numpy_split_list(data: Sequence[T] | Iterable[T], train_ratio: float = 0.8, seed: int = 42) -> Tuple[List[T], List[T]]:
    """Deterministically split an iterable into train/val lists using numpy RNG."""
    data_list = list(data)
    n = len(data_list)
    indices = np.arange(n)

    rng = np.random.default_rng(seed)  # stable, reproducible RNG
    rng.shuffle(indices)

    split_idx = int(n * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train = [data_list[i] for i in train_indices]
    val = [data_list[i] for i in val_indices]
    return train, val


def group_by_prefix(strings: Iterable[str], prefix_len: int = 5) -> List[List[str]]:
    """Group filenames by their prefix (common in ReID datasets)."""
    groups = defaultdict(list)
    for s in strings:
        groups[s[:prefix_len]].append(s)
    return list(groups.values())
