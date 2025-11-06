# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Batch effect simulation for synthetic biomedical datasets.

This module provides functionality to add realistic batch effects
(site differences, instrument variations, recruitment cohorts) to
generated datasets.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def generate_batch_assignments(
    n_samples: int,
    n_batches: int,
    proportions: Sequence[float] | None = None,
    class_labels: NDArray[np.int_] | None = None,
    confounding_strength: float = 0.0,
    random_state: int | None = None,
) -> NDArray[np.int_]:
    """Generate batch assignments for samples.

    Creates batch labels that can be either random or confounded with
    class labels (to simulate recruitment bias, site selection effects, etc.).

    Args:
        n_samples: Number of samples.
        n_batches: Number of batches.
        proportions: Relative batch sizes (None = equal split).
        class_labels: Optional class labels for confounding.
        confounding_strength: Correlation with class (0.0 = random, 1.0 = perfect).
        random_state: Seed for reproducibility.

    Returns:
        Array of batch assignments (integers 0 to n_batches-1).

    Examples:
        >>> # Random assignment
        >>> batches = generate_batch_assignments(100, n_batches=3, random_state=42)
        >>> np.bincount(batches)
        array([33, 33, 34])

        >>> # Confounded with class (simulate recruitment bias)
        >>> y = np.array([0]*50 + [1]*50)
        >>> batches = generate_batch_assignments(
        ...     100, n_batches=2,
        ...     class_labels=y,
        ...     confounding_strength=0.8,
        ...     random_state=42
        ... )
        >>> # Most class 0 in batch 0, most class 1 in batch 1

    Notes:
        - confounding_strength=0.0: Random assignment (no bias)
        - confounding_strength=1.0: Perfect confounding (each class → one batch)
        - confounding_strength=0.5: Moderate bias (70-30 split instead of 50-50)
    """
    rng = np.random.default_rng(random_state)

    if confounding_strength == 0.0 or class_labels is None:
        return _random_batch_assignment(n_samples, n_batches, proportions, rng)
    else:
        return _confounded_batch_assignment(class_labels, n_batches, confounding_strength, proportions, rng)


def _random_batch_assignment(
    n_samples: int,
    n_batches: int,
    proportions: Sequence[float] | None,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Create random batch assignments."""
    if proportions is None:
        # Even split
        base = n_samples // n_batches
        remainder = n_samples % n_batches
        counts = np.array([base + (1 if i < remainder else 0) for i in range(n_batches)], dtype=int)
    else:
        proportions_arr = np.asarray(proportions, dtype=float)
        if not np.isclose(proportions_arr.sum(), 1.0):
            proportions_arr = proportions_arr / proportions_arr.sum()

        counts = (proportions_arr * n_samples).round().astype(int)

        # Fix rounding to match n_samples exactly
        diff = n_samples - counts.sum()
        if diff != 0:
            idx = np.argsort(-proportions_arr)[: abs(diff)]
            counts[idx] += np.sign(diff)

    # Create labels and shuffle
    labels = np.concatenate([np.full(c, i, dtype=int) for i, c in enumerate(counts)])
    rng.shuffle(labels)
    return labels


def _confounded_batch_assignment(
    class_labels: NDArray[np.int_],
    n_batches: int,
    confounding_strength: float,
    proportions: Sequence[float] | None,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Create batch assignments confounded with class labels.

    Strategy:
    - For each class, preferentially assign to certain batches
    - Higher confounding_strength = stronger preference
    - Uses redistribution: probability mass is moved from other batches to preferred batch
    - At strength=1.0: perfect confounding (all samples in preferred batch)
    """
    n_samples = len(class_labels)
    n_classes = int(np.max(class_labels)) + 1

    # Base probabilities for each (class, batch) combination
    if proportions is None:
        base_probs = np.ones((n_classes, n_batches)) / n_batches
    else:
        proportions_arr = np.asarray(proportions, dtype=float)
        proportions_arr = proportions_arr / proportions_arr.sum()
        base_probs = np.tile(proportions_arr, (n_classes, 1))

    # Apply confounding via redistribution
    confounded_probs = base_probs.copy()

    for cls in range(n_classes):
        preferred_batch = cls % n_batches
        base_pref = base_probs[cls, preferred_batch]

        # How much probability mass to add to preferred batch
        boost = confounding_strength * (1.0 - base_pref)

        # How much to subtract from each other batch (redistribute)
        reduction_per_batch = boost / (n_batches - 1)

        # Apply changes
        confounded_probs[cls, preferred_batch] += boost

        # Subtract from all other batches
        for batch_id in range(n_batches):
            if batch_id != preferred_batch:
                confounded_probs[cls, batch_id] -= reduction_per_batch

        # Clip to ensure non-negative (handle floating point precision)
        confounded_probs[cls] = np.clip(confounded_probs[cls], 0.0, 1.0)

        # Renormalize to exactly 1.0 (handle any clipping effects)
        confounded_probs[cls] = confounded_probs[cls] / confounded_probs[cls].sum()

    # Probabilities now sum to 1.0 and are all non-negative

    # Assign batches per class
    batch_assignments = np.zeros(n_samples, dtype=int)
    for cls in range(n_classes):
        mask = class_labels == cls
        n_in_class = np.sum(mask)
        if n_in_class > 0:
            batch_assignments[mask] = rng.choice(n_batches, size=n_in_class, p=confounded_probs[cls])

    return batch_assignments


def apply_batch_effects(
    X: pd.DataFrame | NDArray[np.float64],
    batch_assignments: NDArray[np.int_],
    effect_type: Literal["additive", "multiplicative"] = "additive",
    effect_strength: float = 0.5,
    affected_features: Sequence[int] | Literal["all"] = "all",
    random_state: int | None = None,
) -> tuple[pd.DataFrame | NDArray[np.float64], NDArray[np.float64]]:
    """Apply batch effects to feature matrix.

    Adds systematic differences between batches to simulate:
    - Site-to-site measurement variations
    - Instrument calibration differences
    - Cohort effects (temporal batches)

    Args:
        X: Feature matrix (DataFrame or array).
        batch_assignments: Batch labels for each sample.
        effect_type:
            - "additive": X' = X + b_batch (shifts)
            - "multiplicative": X' = X * (1 + b_batch) (scaling)
        effect_strength: Magnitude (stddev for additive, scale for multiplicative).
        affected_features: Which features to affect ("all" or list of indices).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_affected, batch_intercepts):
            - X_affected: Feature matrix with batch effects applied
            - batch_intercepts: Array of random intercepts drawn for each batch

    Examples:
        >>> from biomedical_data_generator import DatasetConfig, generate_dataset
        >>>
        >>> # Generate clean data
        >>> cfg = DatasetConfig(
        ...     n_samples=100, n_informative=5, n_noise=3,
        ...     n_classes=2, class_counts={0: 50, 1: 50},
        ...     random_state=42
        ... )
        >>> X, y, meta = generate_dataset(cfg, return_dataframe=True)
        >>>
        >>> # Add batch effects
        >>> batches = generate_batch_assignments(len(X), n_batches=3, random_state=42)
        >>> X_batch, intercepts = apply_batch_effects(
        ...     X, batches,
        ...     effect_type="additive",
        ...     effect_strength=0.5,
        ...     random_state=42
        ... )
        >>> print(f"Batch intercepts: {intercepts}")

    Notes:
        - Additive effects: b_g ~ N(0, sigma^2), adds constant offset per batch
        - Multiplicative effects: b_g ~ N(0, sigma^2), scales features by (1 + b_g)
        - Returns copy of X (does not modify in place)
    """
    rng = np.random.default_rng(random_state)

    # Convert to DataFrame if needed for consistent handling
    is_dataframe = isinstance(X, pd.DataFrame)
    X_df: pd.DataFrame | None = None
    feature_names: list[str] | None = None

    if is_dataframe:
        X_df = cast(pd.DataFrame, X)
        X_array = X_df.values.copy()
        feature_names = X_df.columns.tolist()
    else:
        X_array = np.asarray(X, dtype=float).copy()

    n_samples, n_features = X_array.shape
    n_batches = int(np.max(batch_assignments)) + 1

    # Determine which features to affect
    if affected_features == "all":
        feature_indices = list(range(n_features))
    else:
        feature_indices = list(affected_features)

    # Draw random intercepts for each batch
    batch_intercepts = rng.normal(loc=0.0, scale=float(effect_strength), size=n_batches)

    # Apply effects
    for batch_id in range(n_batches):
        mask = batch_assignments == batch_id
        if not np.any(mask):
            continue

        for feat_idx in feature_indices:
            if effect_type == "additive":
                X_array[mask, feat_idx] += batch_intercepts[batch_id]
            elif effect_type == "multiplicative":
                X_array[mask, feat_idx] *= 1.0 + batch_intercepts[batch_id]
            else:
                raise ValueError(f"Unknown effect_type: {effect_type}")

    # Convert back to DataFrame if needed
    X_result: pd.DataFrame | NDArray[np.float64]
    if is_dataframe and X_df is not None and feature_names is not None:
        X_result = pd.DataFrame(X_array, columns=feature_names, index=X_df.index)
    else:
        X_result = X_array

    return X_result, batch_intercepts
