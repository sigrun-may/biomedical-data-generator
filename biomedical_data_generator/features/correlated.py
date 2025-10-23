# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Functions to generate correlated feature clusters simulating biomarker patterns."""

# This module provides a building block to generate
# *correlated* feature clusters.
# It intentionally focuses on the essentials:
#   - Build a *target correlation* matrix R (equicorrelated or Toeplitz).
#   - Factorize R using a robust Cholesky (with diagonal jitter fallback).
#   - Sample X = Z @ L.T with Z ~ N(0, I), so that corr(X) ≈ R (in expectation).
#
# IMPORTANT (for teaching):
# - We do NOT standardize or rescale here. If you need standardization, do it
#   later in generator.py (after assembling all blocks), so responsibilities stay clear.
# - For n very small (e.g., n=30), the *empirical* sample correlations are noisy.
#   The *population* correlation implied by construction is correct, but your
#   finite-sample estimate will vary substantially — this is expected.
# You can extend realism later (copulas, heteroskedasticity, blockwise structures)
# WITHOUT changing this public API by adding optional post-steps elsewhere.

from __future__ import annotations

from typing import Literal, Mapping, Optional

import numpy as np
from numpy.typing import NDArray


CorrelationStructure = Literal["equicorrelated", "toeplitz"]

__all__ = [
    "CorrelationStructure",
    "build_correlation_matrix",
    "sample_cluster",
]


# ============================================================================
# Correlation matrix construction (single source of truth)
# ============================================================================

def build_correlation_matrix(
    n_features: int,
    rho: float,
    structure: CorrelationStructure,
) -> NDArray[np.float64]:
    """Build a correlation matrix with the requested structure.

    Two supported patterns:

    1) Equicorrelated (compound symmetry)
       R = (1 - rho) * I + rho * J
       Positive definite iff  -1/(p-1) < rho < 1  for p = n_features (strict).
       Intuition: Every pair of features shares the same correlation rho.

    2) Toeplitz (AR(1)-like)
       R_ij = rho ** |i - j|
       Positive definite for |rho| < 1 (strict).
       Intuition: Features are ordered; correlation decays with distance.

    Args:
        n_features: Number of columns p in the cluster (p > 0).
        rho: Correlation strength parameter (validated per structure).
        structure: Either "equicorrelated" or "toeplitz".

    Returns:
        Correlation matrix of shape (p, p), dtype float64.

    Raises:
        ValueError: If n_features <= 0 or if rho violates PD constraints.
    """
    if n_features <= 0:
        raise ValueError(f"n_features must be positive, got {n_features}")

    if structure == "equicorrelated":
        # PD condition: -1/(p-1) < rho < 1 (strict).
        # For p=1, the lower bound is irrelevant; any rho<1 will produce [1].
        lower_bound = -1.0 / (n_features - 1) if n_features > 1 else -np.inf
        if not (lower_bound < rho < 1.0):
            raise ValueError(
                f"Invalid rho={rho} for equicorrelated with n_features={n_features}; "
                f"require {lower_bound:.6f} < rho < 1."
            )
        identity = np.eye(n_features, dtype=np.float64)
        ones = np.ones((n_features, n_features), dtype=np.float64)
        return (1.0 - rho) * identity + rho * ones

    if structure == "toeplitz":
        # PD condition: |rho| < 1 (strict).
        if not (-1.0 < rho < 1.0):
            raise ValueError(f"Invalid rho={rho} for toeplitz; require |rho| < 1.")
        indices = np.arange(n_features, dtype=np.int64)
        distances = np.abs(indices[:, None] - indices[None, :])
        corr_matrix = np.power(rho, distances, dtype=np.float64)
        # Fill the diagonal with exact ones for numerical robustness.
        np.fill_diagonal(corr_matrix, 1.0)
        return corr_matrix

    raise ValueError(f"Unknown structure={structure!r}")


# ============================================================================
# Robust Cholesky factorization with diagonal jitter fallback
# ============================================================================

def _cholesky_with_jitter(
    corr_matrix: NDArray[np.float64],
    *,
    max_tries: int = 6,
    initial_jitter: float = 1e-12,
    growth: float = 10.0,
) -> NDArray[np.float64]:
    """Compute a Cholesky factor with diagonal jitter fallback.

    Why this helper?
      In finite-precision arithmetic (float64), tiny negative eigenvalues can
      appear due to rounding, even if the intended matrix is PD. We try a clean
      Cholesky first; if it fails, we add a tiny "jitter" (ε * I) and retry.

    Args:
        corr_matrix: Target correlation matrix R (p x p), symmetric with diag≈1.
        max_tries: Maximum number of jitter escalations.
        initial_jitter: Starting jitter ε for the first retry.
        growth: Multiplicative factor to increase jitter each failed attempt.

    Returns:
        Lower-triangular matrix L such that L @ L.T ≈ corr_matrix.

    Raises:
        np.linalg.LinAlgError: If factorization fails after all attempts.
    """
    try:
        return np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        pass  # Fall back to jittered attempts.

    p = corr_matrix.shape[0]
    identity = np.eye(p, dtype=np.float64)
    jitter = initial_jitter

    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(corr_matrix + jitter * identity)
        except np.linalg.LinAlgError:
            jitter *= growth

    # Let NumPy raise a clean error using the original matrix.
    np.linalg.cholesky(corr_matrix)  # will raise


# ============================================================================
# Public sampler
# ============================================================================

def sample_cluster(
    n_samples: int,
    n_features: int,
    rng: np.random.Generator,
    *,
    structure: CorrelationStructure = "equicorrelated",
    rho: Optional[float] = None,
    class_labels: Optional[NDArray[np.int64]] = None,
    class_rho: Optional[Mapping[int, float]] = None,
    baseline_rho: float = 0.0,
) -> NDArray[np.float64]:
    """Sample a correlated feature block (global or class-specific).

    Behavior:
      - Global mode:
            If `class_labels` is None, a single correlation matrix R is used for
            all samples. This requires a scalar `rho`.
      - Class-specific mode:
            If `class_labels` is provided (int64 array of shape (n_samples,)),
            each class k can have its own correlation strength ρ_k. You pass
            overrides via `class_rho={k: rho_k}`; classes not listed in the
            mapping use `baseline_rho`. The *structure* is the same for all.

    No standardization or scaling is performed here — this keeps the sampler
    pure and easy to reason about. If you want class separation (mean shifts),
    apply them later (e.g., in generator.py). A constant mean shift per class
    DOES NOT change within-class correlations.

    Args:
        n_samples: Number of rows to generate.
        n_features: Number of columns in this cluster.
        rng: NumPy random Generator (use a shared one for reproducibility).
        structure: Correlation structure ("equicorrelated" or "toeplitz").
        rho: Global correlation strength (required in global mode).
        class_labels: Optional labels (int64) of shape (n_samples,).
        class_rho: Optional mapping {class_label -> rho} for overrides.
        baseline_rho: Fallback rho if a class has no override (default 0.0).

    Returns:
        Array X of shape (n_samples, n_features), dtype float64.

    Raises:
        ValueError: If arguments are inconsistent (e.g., missing rho in global
            mode, label length mismatch, invalid ρ per structure).
    """
    # -------------------------------
    # Class-specific mode (if labels)
    # -------------------------------
    if class_labels is not None:
        if class_labels.shape[0] != n_samples:
            raise ValueError(
                f"class_labels has length {class_labels.shape[0]} but n_samples={n_samples}"
            )

        overrides = class_rho or {}
        X = np.empty((n_samples, n_features), dtype=np.float64)

        # Sample per class so each group can use its own rho, but the same structure.
        unique_classes = np.unique(class_labels)
        for cls in unique_classes:
            class_mask = (class_labels == cls)
            n_in_class = int(class_mask.sum())
            if n_in_class == 0:
                continue

            rho_for_class = float(overrides.get(int(cls), baseline_rho))
            R_class = build_correlation_matrix(n_features, rho_for_class, structure)
            L_class = _cholesky_with_jitter(R_class)

            standard_normal_block = rng.standard_normal(size=(n_in_class, n_features))
            X[class_mask] = standard_normal_block @ L_class.T

        return X

    # -------------------------------
    # Global mode (no labels provided)
    # -------------------------------
    if rho is None:
        raise ValueError(
            "Global mode requires `rho` when no `class_labels` are provided."
        )

    R_global = build_correlation_matrix(n_features, rho, structure)
    L_global = _cholesky_with_jitter(R_global)

    standard_normal = rng.standard_normal(size=(n_samples, n_features))
    return standard_normal @ L_global.T

