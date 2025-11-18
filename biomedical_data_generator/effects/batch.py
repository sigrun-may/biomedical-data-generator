# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Batch effect simulation for synthetic biomedical datasets.

This module provides functionality to add realistic batch effects
(site differences, instrument variations, recruitment cohorts) to
generated datasets.

Public API:
    High-level (recommended):
        - apply_batch_effects_from_config: Orchestrate assignment + effects
        - generate_batch_assignments: Unified batch assignment (random or confounded)
        - apply_batch_effects: Apply systematic feature differences

    Low-level (for experimentation):
        - random_batch_assignment: Create independent batch labels
        - confounded_batch_assignment: Create class-correlated batch labels

Typical usage (high-level):
    >>> from batch import apply_batch_effects_from_config
    >>> X_batch, batches, batch_effects = apply_batch_effects_from_config(
    ...     X=X, y=y, batch_config=cfg.batch,
    ...     informative_indices=inf_idx, rng=rng
    ... )

Educational usage (low-level):
    >>> # Experiment with random vs. confounded assignments
    >>> rng = np.random.default_rng(42)
    >>> random_batches = random_batch_assignment(100, n_batches=3, rng=rng)
    >>> confounded_batches = confounded_batch_assignment(
    ...     y, n_batches=3, confounding_with_class=0.8, rng=rng
    ... )

Reproducibility:
    All functions accept np.random.Generator objects to ensure proper
    state propagation. The top-level code (generator.py) creates ONE
    generator and passes it through the call chain. This ensures:
    - No duplicate random sequences
    - Perfect reproducibility from config.random_state
    - Proper state evolution across all operations
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import ArrayLike


__all__ = [
    "generate_batch_assignments",
    "random_batch_assignment",
    "confounded_batch_assignment",
    "apply_batch_effects",
    "apply_batch_effects_from_config",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_proportions(
    proportions: Sequence[float],
    n_batches: int,
) -> NDArray[np.float64]:
    """Normalize a proportions vector to length-n_batches and sum 1.

    Raises:
        ValueError: if length mismatch, negative entries, or non-positive sum.
    """
    proportions_arr = np.asarray(proportions, dtype=float)

    if proportions_arr.shape[0] != n_batches:
        raise ValueError("len(proportions) must equal n_batches")

    if np.any(proportions_arr < 0):
        raise ValueError("proportions must be non-negative")

    total = proportions_arr.sum()
    if total <= 0:
        raise ValueError("proportions must sum to a positive value")

    return proportions_arr / total


# ---------------------------------------------------------------------------
# Batch assignment
# ---------------------------------------------------------------------------


def generate_batch_assignments(
    n_samples: int,
    n_batches: int,
    rng: np.random.Generator,
    proportions: Sequence[float] | None = None,
    class_labels: ArrayLike | None = None,
    confounding_with_class: float = 0.0,
) -> NDArray[np.int_]:
    """Generate batch assignments for samples.

    Unified entry point for creating batch labels. Automatically handles
    both random and confounded assignments based on the confounding_with_class
    parameter.

    This is the recommended function for batch assignment. For direct access
    to the underlying implementations, see:
        - random_batch_assignment() for independent assignments
        - confounded_batch_assignment() for class-correlated assignments

    Args:
        n_samples: Number of samples.
        n_batches: Number of batches (must be >= 1).
        rng: NumPy random generator (for reproducibility).
        proportions: Relative batch sizes (None = equal split).
        class_labels: Optional class labels for confounding. May be any
            array-like of shape (n_samples,) with labels that np.unique can
            handle (e.g. integers, strings). Only used if
            confounding_with_class > 0.
        confounding_with_class: Correlation with class
            (0.0 = random, 1.0 = strong preference), must be in [0, 1].

    Returns:
        Array of batch assignments (integers 0 to n_batches-1).

    Examples:
        >>> rng = np.random.default_rng(42)
        >>>
        >>> # Random assignment
        >>> batches = generate_batch_assignments(100, n_batches=3, rng=rng)
        >>> np.bincount(batches)
        array([33, 33, 34])
        >>>
        >>> # Confounded with class (simulate recruitment bias)
        >>> y = np.array([0]*50 + [1]*50)
        >>> batches = generate_batch_assignments(
        ...     n_samples=100,
        ...     n_batches=2,
        ...     rng=rng,
        ...     class_labels=y,
        ...     confounding_with_class=0.8,
        ... )

    Notes:
        - confounding_with_class=0.0: Random assignment (no bias)
        - confounding_with_class=1.0: Strong preference (each class → preferred batch)
        - For 2 equal-sized batches, confounding_with_class=0.5 yields
          approx. a 75/25 class split towards the preferred batch.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if n_batches <= 0:
        raise ValueError("n_batches must be >= 1")
    if not (0.0 <= confounding_with_class <= 1.0):
        raise ValueError("confounding_with_class must be in [0, 1]")

    # Trivial case: only one batch → all zeros (no confounding possible)
    if n_batches == 1:
        return np.zeros(n_samples, dtype=int)

    if class_labels is not None:
        labels_arr = np.asarray(class_labels)
        if labels_arr.shape[0] != n_samples:
            raise ValueError("class_labels must have length n_samples")
    else:
        labels_arr = None

    if confounding_with_class == 0.0 or labels_arr is None:
        return random_batch_assignment(
            n_samples=n_samples,
            n_batches=n_batches,
            rng=rng,
            proportions=proportions,
        )
    else:
        return confounded_batch_assignment(
            class_labels=labels_arr,
            n_batches=n_batches,
            confounding_with_class=confounding_with_class,
            rng=rng,
            proportions=proportions,
        )


def random_batch_assignment(
    n_samples: int,
    n_batches: int,
    rng: np.random.Generator,
    proportions: Sequence[float] | None = None,
) -> NDArray[np.int_]:
    """Create random batch assignments without confounding.

    Direct function for creating batch labels with specified proportions
    but no correlation with class labels.

    For most use cases, prefer generate_batch_assignments() which provides
    a unified interface for both random and confounded assignments.

    Args:
        n_samples: Number of samples to assign.
        n_batches: Number of distinct batches (must be >= 1).
        rng: NumPy random generator (for reproducibility).
        proportions: Relative batch sizes (None = equal split).

    Returns:
        Array of batch assignments (integers 0 to n_batches-1).

    Examples:
        >>> rng = np.random.default_rng(42)
        >>> batches = random_batch_assignment(100, n_batches=3, rng=rng)
        >>> np.bincount(batches)
        array([33, 33, 34])
        >>>
        >>> # Unequal proportions
        >>> batches = random_batch_assignment(
        ...     100, n_batches=3, rng=rng, proportions=[0.5, 0.3, 0.2]
        ... )

    Notes:
        - Generator state advances during shuffling.
        - Multiple calls with the same generator produce different results.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if n_batches <= 0:
        raise ValueError("n_batches must be >= 1")

    if proportions is None:
        # Even split (integer counts with remainder distributed over first batches)
        base = n_samples // n_batches
        remainder = n_samples % n_batches
        counts = np.array(
            [base + (1 if i < remainder else 0) for i in range(n_batches)],
            dtype=int,
        )
    else:
        proportions_arr = _normalize_proportions(proportions, n_batches)
        counts = np.round(proportions_arr * n_samples).astype(int)

        # Fix rounding to match n_samples exactly
        diff = n_samples - int(counts.sum())
        if diff != 0:
            # Adjust the batches with largest target proportions first
            idx = np.argsort(-proportions_arr)[: abs(diff)]
            counts[idx] += int(np.sign(diff))

    # Create labels and shuffle
    labels = np.concatenate([np.full(c, i, dtype=int) for i, c in enumerate(counts) if c > 0])
    # In pathological rounding cases we might end up slightly short/long; guard here
    if labels.shape[0] != n_samples:
        # Simple fallback: sample with replacement from batches according to proportions
        if proportions is None:
            p = np.full(n_batches, 1.0 / n_batches)
        else:
            p = _normalize_proportions(proportions, n_batches)
        labels = rng.choice(n_batches, size=n_samples, p=p)

    rng.shuffle(labels)
    return labels.astype(int)


def confounded_batch_assignment(
    class_labels: ArrayLike,
    n_batches: int,
    confounding_with_class: float,
    rng: np.random.Generator,
    proportions: Sequence[float] | None = None,
) -> NDArray[np.int_]:
    """Create batch assignments confounded with class labels.

    Direct function for creating batch labels that are correlated with
    class membership, simulating recruitment bias or site selection effects.

    For most use cases, prefer generate_batch_assignments() which provides
    a unified interface for both random and confounded assignments.

    Strategy:
        - Use np.unique to derive K distinct classes (works for int and str).
        - For each class index k (0..K-1), preferentially assign to a
          "preferred" batch (k % n_batches).
        - Higher confounding_with_class = stronger preference for the
          preferred batch.
        - Uses redistribution: probability mass is moved from other batches
          to the preferred batch.
        - At confounding_with_class=1.0: strong preference (all samples of a
          class tend to fall into its preferred batch, subject to randomness).

    Args:
        class_labels: Class membership for each sample. May be integer or
            string labels; only relative grouping matters.
        n_batches: Number of distinct batches (must be >= 2).
        confounding_with_class: Degree of correlation (0.0 = random assignment,
            1.0 = strong preference), must be in [0, 1].
        rng: NumPy random generator (for reproducibility).
        proportions: Relative batch sizes (None = equal split).

    Returns:
        Array of batch assignments (integers 0 to n_batches-1).

    Examples:
        >>> rng = np.random.default_rng(42)
        >>> y = np.array(["control"] * 50 + ["case"] * 50)
        >>> batches = confounded_batch_assignment(
        ...     y, n_batches=2, confounding_with_class=0.8, rng=rng
        ... )
        >>> # Most "control" samples will be in batch 0,
        >>> # most "case" samples will be in batch 1.

    Notes:
        - confounding_with_class=0.0 would reduce to random assignment;
          use generate_batch_assignments for that branch.
        - For 2 equal-sized batches, confounding_with_class=0.5 yields
          approx. a 75/25 preference towards the class's preferred batch.
    """
    labels = np.asarray(class_labels)
    if labels.ndim != 1:
        labels = labels.ravel()

    n_samples = labels.shape[0]
    if n_samples == 0:
        raise ValueError("class_labels must contain at least one sample")

    if n_batches <= 1:
        raise ValueError("confounded_batch_assignment requires n_batches >= 2")

    if not (0.0 <= confounding_with_class <= 1.0):
        raise ValueError("confounding_with_class must be in [0, 1]")

    # Map arbitrary labels → integer class indices 0..K-1
    unique_classes, class_indices = np.unique(labels, return_inverse=True)
    n_classes = unique_classes.shape[0]

    # Base probabilities for each (class, batch) combination
    if proportions is None:
        base_probs = np.full((n_classes, n_batches), 1.0 / n_batches, dtype=float)
        base_proportions = np.full(n_batches, 1.0 / n_batches, dtype=float)
    else:
        base_proportions = _normalize_proportions(proportions, n_batches)
        base_probs = np.tile(base_proportions, (n_classes, 1))

    # Apply confounding via redistribution
    confounded_probs = base_probs.copy()

    for cls_idx in range(n_classes):
        preferred_batch = cls_idx % n_batches
        base_pref = base_probs[cls_idx, preferred_batch]

        # Amount of probability mass to add to preferred batch
        boost = confounding_with_class * (1.0 - base_pref)

        if boost <= 0:
            # Nothing to do for this class (e.g., confounding_with_class == 0)
            continue

        # How much to subtract from each other batch (redistribute)
        reduction_per_batch = boost / (n_batches - 1)

        # Apply changes: add to preferred, subtract from others
        confounded_probs[cls_idx, preferred_batch] += boost
        for batch_id in range(n_batches):
            if batch_id != preferred_batch:
                confounded_probs[cls_idx, batch_id] -= reduction_per_batch

        # Clip to ensure non-negative (handle floating point precision)
        confounded_probs[cls_idx] = np.clip(confounded_probs[cls_idx], 0.0, 1.0)

        # Renormalize to exactly 1.0 (handle any clipping effects)
        row_sum = confounded_probs[cls_idx].sum()
        if row_sum <= 0:
            # Fallback: revert to base probabilities for this class
            confounded_probs[cls_idx] = base_probs[cls_idx]
        else:
            confounded_probs[cls_idx] /= row_sum

    # Assign batches per class index (0..K-1)
    batch_assignments = np.empty(n_samples, dtype=int)
    for cls_idx in range(n_classes):
        mask = class_indices == cls_idx
        n_in_class = int(mask.sum())
        if n_in_class == 0:
            continue
        batch_assignments[mask] = rng.choice(
            n_batches,
            size=n_in_class,
            p=confounded_probs[cls_idx],
        )

    return batch_assignments.astype(int)


# ---------------------------------------------------------------------------
# Applying batch effects to feature matrices
# ---------------------------------------------------------------------------
def apply_batch_effects(
    X: pd.DataFrame | NDArray[np.float64],
    batch_assignments: NDArray[np.int_],
    rng: np.random.Generator,
    effect_type: Literal["additive", "multiplicative"] = "additive",
    effect_strength: float = 0.5,
    affected_features: Sequence[int] | Literal["all"] = "all",
) -> tuple[pd.DataFrame | NDArray[np.float64], NDArray[np.float64]]:
    """Apply batch effects to a feature matrix.

    Apply systematic batch effects to specified features in the dataset,
    simulating site differences or instrument variations.

    Args:
        X: Feature matrix (DataFrame or array).
        batch_assignments: Array of batch assignments per sample.
        rng: NumPy random generator (for reproducibility).
        effect_type: "additive" or "multiplicative" batch effects.
        effect_strength: Standard deviation of batch effects.
        affected_features: "all" or list of feature indices to affect.

    Returns:
        Tuple of (X_affected, batch_effects):
            - X_affected: Feature matrix with batch effects applied
            - batch_effects: Random effects drawn per batch
    Examples:
        >>> rng = np.random.default_rng(42)
        >>> X = np.random.normal(size=(100, 5))
        >>> batches = np.random.randint(0, 3, size=100)
        >>>
        >>> X_batch, batch_effects = apply_batch_effects(
        ...     X, batches, rng,
        ...     effect_type="additive",
        ...     effect_strength=0.5,
        ...     affected_features=[0, 2]
        ... )

    Notes:
        - Generator state advances during effect sampling.
    """
    # Convert to DataFrame if needed for consistent handling
    is_dataframe = isinstance(X, pd.DataFrame)
    X_df: pd.DataFrame | None = None
    feature_names: list[str] | None = None

    if is_dataframe:
        X_df = cast(pd.DataFrame, X)
        X_array = X_df.to_numpy(copy=True, dtype=float)
        feature_names = X_df.columns.tolist()
    else:
        X_array = np.asarray(X, dtype=float).copy()

    n_samples, n_features = X_array.shape

    batch_assignments = np.asarray(batch_assignments, dtype=int)
    if batch_assignments.shape[0] != n_samples:
        raise ValueError("batch_assignments must have length n_samples")
    if np.any(batch_assignments < 0):
        raise ValueError("batch_assignments must be non-negative integers")

    n_batches = int(batch_assignments.max()) + 1

    # Determine which features to affect
    if affected_features == "all":
        feature_indices = np.arange(n_features, dtype=int)
    else:
        feature_indices = np.asarray(list(affected_features), dtype=int)
        if feature_indices.ndim != 1:
            feature_indices = feature_indices.ravel()
        if feature_indices.size == 0:
            # Nothing to do; return copy of X and an empty effects array
            batch_effects = np.zeros(n_batches, dtype=float)
            if is_dataframe and X_df is not None and feature_names is not None:
                return pd.DataFrame(X_array, columns=feature_names, index=X_df.index), batch_effects
            else:
                return X_array, batch_effects

        if np.any((feature_indices < 0) | (feature_indices >= n_features)):
            raise IndexError("affected_features contains indices out of range")

    # Draw random effects for each batch
    batch_effects = rng.normal(loc=0.0, scale=float(effect_strength), size=n_batches)

    # Map per-batch effects to per-sample vector
    effects_per_sample = batch_effects[batch_assignments]  # shape (n_samples,)

    # Apply effects in a vectorized way
    if effect_type == "additive":
        X_array[:, feature_indices] += effects_per_sample[:, None]
    elif effect_type == "multiplicative":
        X_array[:, feature_indices] *= (1.0 + effects_per_sample)[:, None]
    else:
        raise ValueError(f"Unknown effect_type: {effect_type}")

    # Convert back to DataFrame if needed
    if is_dataframe and X_df is not None and feature_names is not None:
        X_result: pd.DataFrame | NDArray[np.float64] = pd.DataFrame(X_array, columns=feature_names, index=X_df.index)
    else:
        X_result = X_array

    return X_result, batch_effects


def apply_batch_effects_from_config(
    X: pd.DataFrame | NDArray[np.float64],
    y: NDArray[np.int_] | ArrayLike,
    batch_config: Any,
    informative_indices: Sequence[int],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame | NDArray[np.float64], NDArray[np.int_], NDArray[np.float64]]:
    """Apply batch effects based on a configuration object.

    High-level orchestration function that handles batch assignment and
    effect application in a single call. Simplifies generator code by
    encapsulating all batch-related logic.

    Args:
        X: Feature matrix (DataFrame or array).
        y: Class labels (for potential confounding). May be any array-like
           accepted by np.unique (e.g. integers or strings).
        batch_config: Configuration object with attributes:
            - n_batches: Number of batches (>= 1)
            - proportions: Optional batch size proportions
            - confounding_with_class: Degree of class-batch correlation in [0, 1]
            - effect_type: "additive" or "multiplicative"
            - effect_strength: Magnitude of batch effects
            - affected_features: "all", "informative", or list of indices
        informative_indices: Indices of informative features (used if
            affected_features="informative").
        rng: NumPy random generator (for reproducibility).

    Returns:
        Tuple of (X_affected, batch_labels, batch_effects):
            - X_affected: Feature matrix with batch effects applied
            - batch_labels: Array of batch assignments per sample
            - batch_effects: Random effects drawn per batch

    Examples:
        >>> from biomedical_data_generator import DatasetConfig, generate_dataset
        >>> from batch import apply_batch_effects_from_config
        >>> import numpy as np
        >>>
        >>> cfg = DatasetConfig(
        ...     n_samples=100, n_informative=5, n_noise=3,
        ...     batch=BatchConfig(n_batches=3, effect_strength=0.5)
        ... )
        >>> X, y, meta = generate_dataset(cfg, return_dataframe=True)
        >>> rng = np.random.default_rng(42)
        >>> X_batch, batches, batch_effects = apply_batch_effects_from_config(
        ...     X, y, cfg.batch, meta.informative_idx, rng=rng
        ... )

    Notes:
        - Automatically handles confounding when confounding_with_class > 0.
        - Maps "informative" to actual feature indices.
        - Centralizes all batch logic in one place.
        - Generator state advances through assignment and effect application.
    """
    y_arr = np.asarray(y)
    n_samples = y_arr.shape[0]

    # Generate batch assignments (handles confounding internally)
    batch_labels = generate_batch_assignments(
        n_samples=n_samples,
        n_batches=batch_config.n_batches,
        rng=rng,
        proportions=batch_config.proportions,
        class_labels=y_arr if batch_config.confounding_with_class > 0.0 else None,
        confounding_with_class=batch_config.confounding_with_class,
    )

    # Resolve affected features based on config
    if batch_config.affected_features == "informative":
        affected: Sequence[int] | Literal["all"] = list(informative_indices)
    elif batch_config.affected_features == "all":
        affected = "all"
    else:
        # Custom list of indices
        affected = batch_config.affected_features

    # Apply batch effects to feature matrix
    X_affected, batch_effects = apply_batch_effects(
        X=X,
        batch_assignments=batch_labels,
        rng=rng,
        effect_type=batch_config.effect_type,
        effect_strength=batch_config.effect_strength,
        affected_features=affected,
    )

    return X_affected, batch_labels, batch_effects
