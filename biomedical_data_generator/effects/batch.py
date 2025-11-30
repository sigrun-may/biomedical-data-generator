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
    ...     x=X, y=y, batch_config=cfg.batch_effects,
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
    effect_granularity: Literal["per_feature", "scalar"] = "per_feature",
) -> tuple[pd.DataFrame | NDArray[np.float64], NDArray[np.float64]]:
    """Apply batch effects to a feature matrix.

    Apply systematic batch effects to specified features in the dataset,
    simulating site differences or instrument variations.

    Args:
        X: Feature matrix (DataFrame or array).
        batch_assignments: Array of batch assignments per sample.
        rng:
            NumPy random number generator used to sample batch effects.
        effect_type:
            Type of batch effect to apply:
            - ``"additive"``: Adds a batch-specific shift to affected features.
            - ``"multiplicative"``: Multiplies affected features by a batch-specific
              scaling factor around 1.0.
        effect_strength:
            Standard deviation of the batch-effect distribution. Larger values
            generate stronger perturbations. Must be non-negative.
        affected_features:
            Defines which features are affected:
            - ``"all"``: Apply batch effects to all features.
            - Sequence[int]: Apply effects only to the listed feature indices.
        effect_granularity:
            Controls whether effects vary across features:
            - ``"per_feature"`` (default): Draws effects with shape
              ``(n_batches, n_affected_features)``, so features within the same batch
              differ in their batch-specific effect.
            - ``"scalar"``: Draws a single scalar per batch and applies it uniformly
              across all affected features. This corresponds to a global per-batch
              offset (additive) or global scaling factor (multiplicative).

    Returns:
        Tuple[pd.DataFrame | np.ndarray, np.ndarray]:
            A tuple ``(X_affected, batch_effects)`` where:

            - **X_affected**: The feature matrix with batch effects applied.
              The output type matches the input type (DataFrame or ndarray).

            - **batch_effects**: An array of length ``n_batches`` summarizing
              the effect applied in each batch:
                * For ``effect_granularity="scalar"``, these are the exact additive
                  shifts (additive mode) or multiplicative factors minus 1.0
                  (multiplicative mode) drawn for each batch.
                * For ``"per_feature"``, these values are the mean additive effect
                  or mean multiplicative deviation from 1.0 across all affected
                  features in each batch. This provides a compact summary even though
                  the full per-feature effects vary within batches.

    Raises:
        ValueError:
            If ``batch_assignments`` has the wrong shape, contains negative values,
            or ``effect_granularity`` is not one of the allowed strings.
        IndexError:
            If ``affected_features`` contains indices outside the valid feature range.

    Notes:
        - The generator state ``rng`` is advanced during sampling.
        - Additive effects are sampled from ``Normal(0, effect_strength)``.
        - Multiplicative effects are sampled from ``1 + Normal(0, effect_strength)``.
        - For DataFrame input, column names and index are preserved in the output.

    Examples:
        >>> rng = np.random.default_rng(42)
        >>> x = np.random.normal(size=(100, 5))
        >>> batches = np.random.randint(0, 3, size=100)
        >>>
        >>> X_batch, batch_effects = apply_batch_effects(
        ...     x, batches, rng,
        ...     effect_type="additive",
        ...     effect_strength=0.5,
        ...     affected_features=[0, 2],
        ...     effect_granularity="scalar",
        ... )
        >>> X_batch.shape
        (100, 5)
        >>> batch_effects.shape
        (3,)
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

    # Quick path: no effect
    if float(effect_strength) == 0.0:
        batch_effects = np.zeros(n_batches, dtype=float)
        if is_dataframe and X_df is not None and feature_names is not None:
            return pd.DataFrame(X_array, columns=feature_names, index=X_df.index), batch_effects
        return X_array, batch_effects

    m = feature_indices.size

    if effect_granularity not in ("per_feature", "scalar"):
        raise ValueError("effect_granularity must be 'per_feature' or 'scalar'")

    # Draw and apply effects according to granularity
    if effect_type == "additive":
        if effect_granularity == "per_feature":
            # shape (n_batches, m)
            effects = rng.normal(loc=0.0, scale=float(effect_strength), size=(n_batches, m))
            per_sample_effects = effects[batch_assignments]  # (n_samples, m)
            X_array[:, feature_indices] += per_sample_effects
            batch_effects = effects.mean(axis=1).astype(float)
        else:  # scalar per batch
            effects = rng.normal(loc=0.0, scale=float(effect_strength), size=n_batches)  # (n_batches,)
            per_sample = effects[batch_assignments]  # (n_samples,)
            X_array[:, feature_indices] += per_sample[:, None]  # broadcast across features
            batch_effects = effects.astype(float)

    elif effect_type == "multiplicative":
        if effect_granularity == "per_feature":
            factors = 1.0 + rng.normal(loc=0.0, scale=float(effect_strength), size=(n_batches, m))
            per_sample_factors = factors[batch_assignments]  # (n_samples, m)
            X_array[:, feature_indices] *= per_sample_factors
            batch_effects = (factors - 1.0).mean(axis=1).astype(float)
        else:  # scalar per batch
            factors = 1.0 + rng.normal(loc=0.0, scale=float(effect_strength), size=n_batches)  # (n_batches,)
            per_sample = factors[batch_assignments]
            X_array[:, feature_indices] *= per_sample[:, None]
            batch_effects = (factors - 1.0).astype(float)
    else:
        raise ValueError(f"Unknown effect_type: {effect_type}")

    # Convert back to DataFrame if needed
    if is_dataframe and X_df is not None and feature_names is not None:
        X_result: pd.DataFrame | NDArray[np.float64] = pd.DataFrame(X_array, columns=feature_names, index=X_df.index)
    else:
        X_result = X_array

    return X_result, batch_effects


def apply_batch_effects_from_config(
    x: pd.DataFrame | NDArray[np.float64],
    y: NDArray[np.int_] | ArrayLike,
    batch_config: Any,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame | NDArray[np.float64], NDArray[np.int_], NDArray[np.float64]]:
    """Apply batch effects based on a configuration object.

    High-level orchestration function that handles batch assignment and
    effect application in a single call.

    Args:
        x: Feature matrix (DataFrame or array).
        y: Class labels (for potential confounding). May be any array-like
           accepted by np.unique (e.g. integers or strings).
        batch_config: Configuration object with attributes:
            - n_batches: Number of batches (>= 1)
            - proportions: Optional batch size proportions
            - confounding_with_class: Degree of class-batch correlation in [0, 1]
            - effect_type: "additive" or "multiplicative"
            - effect_strength: Magnitude of batch effects
            - affected_features: "all", "informative", or list of indices
        rng: NumPy random generator (for reproducibility).

    Returns:
        Tuple of (X_affected, batch_labels, batch_effects):
            - X_affected: Feature matrix with batch effects applied
            - batch_labels: Array of batch assignments per sample
            - batch_effects: Random effects drawn per batch

    Notes:
        - Automatically handles confounding when confounding_with_class > 0.
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

    # Apply batch effects to feature matrix
    X_affected, batch_effects = apply_batch_effects(
        X=x,
        batch_assignments=batch_labels,
        rng=rng,
        effect_type=batch_config.effect_type,
        effect_strength=batch_config.effect_strength,
        affected_features=batch_config.affected_features,
    )

    return X_affected, batch_labels, batch_effects
