# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Functions to generate correlated feature clusters simulating biomarker patterns.

IMPORTANT:
- We do NOT standardize or rescale here. If you need standardization, do it
  later in generator.py (after assembling all blocks), so responsibilities stay clear.
- For n very small (e.g., n=30), the *empirical* sample correlations are noisy.
  The *population* correlation implied by construction is correct, but your
  finite-sample estimate will vary substantially — this is expected.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from biomedical_data_generator import CorrClusterConfig, DatasetConfig

CorrelationStructure = Literal["equicorrelated", "toeplitz"]

__all__ = [
    "CorrelationStructure",
    "build_correlation_matrix",
    "sample_correlated_cluster",
    "sample_all_correlated_clusters",
]


# ============================================================================
# Correlation matrix construction (single source of truth)
# ============================================================================
def build_correlation_matrix(
    n_features: int,
    correlation: float,
    structure: CorrelationStructure,
) -> NDArray[np.float64]:
    """Build a correlation matrix with the requested structure.

    Two supported patterns:

    1) Equicorrelated (compound symmetry)
       R = (1 - ρ) * I + ρ * J
       Positive definite iff  -1/(p-1) < ρ < 1  for p = n_features (strict).
       Intuition: Every pair of features shares the same correlation ρ.

    2) Toeplitz (AR(1)-like)
       R_ij = ρ ** |i - j|
       Positive definite for |ρ| < 1 (strict).
       Intuition: Features are ordered; correlation decays with distance.

    Args:
        n_features:
            Number of columns p in the cluster (p > 0).
        correlation:
            Correlation strength parameter ρ (validated per structure).
        structure:
            Either "equicorrelated" or "toeplitz".

    Returns:
        Correlation matrix of shape (p, p), dtype float64.

    Raises:
        ValueError:
            If n_features <= 0 or if correlation violates PD constraints.
    """
    if n_features <= 0:
        raise ValueError(f"n_features must be positive, got {n_features}")

    if structure == "equicorrelated":
        # PD condition: -1/(p-1) < rho < 1 (strict).
        # For p=1, the lower bound is irrelevant; any rho<1 will produce [1].
        lower_bound = -1.0 / (n_features - 1) if n_features > 1 else -np.inf
        if not (lower_bound < correlation < 1.0):
            raise ValueError(
                f"Invalid correlation={correlation} for equicorrelated with "
                f"n_features={n_features}; require {lower_bound:.6f} < correlation < 1."
            )
        identity = np.eye(n_features, dtype=np.float64)
        ones = np.ones((n_features, n_features), dtype=np.float64)
        return (1.0 - correlation) * identity + correlation * ones

    if structure == "toeplitz":
        # PD condition: |correlation| < 1 (strict).
        if not (-1.0 < correlation < 1.0):
            raise ValueError(
                f"Invalid correlation={correlation} for toeplitz; require |correlation| < 1."
            )
        indices = np.arange(n_features, dtype=np.int64)
        distances = np.abs(indices[:, None] - indices[None, :])
        corr_matrix = np.power(correlation, distances, dtype=np.float64)
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
        corr_matrix:
            Target correlation matrix R (p x p), symmetric with diag≈1.
        max_tries:
            Maximum number of jitter escalations.
        initial_jitter:
            Starting jitter ε for the first retry.
        growth:
            Multiplicative factor to increase jitter each failed attempt.

    Returns:
        Lower-triangular matrix L such that L @ L.T ≈ corr_matrix.

    Raises:
        np.linalg.LinAlgError:
            If factorization fails after all attempts.
    """
    try:
        return np.asarray(np.linalg.cholesky(corr_matrix), dtype=np.float64)
    except np.linalg.LinAlgError:
        pass  # Fall back to jittered attempts.

    p = corr_matrix.shape[0]
    identity = np.eye(p, dtype=np.float64)
    jitter = initial_jitter

    for _ in range(max_tries):
        try:
            return np.asarray(
                np.linalg.cholesky(corr_matrix + jitter * identity),
                dtype=np.float64,
            )
        except np.linalg.LinAlgError:
            jitter *= growth

    # Let NumPy raise a clean error using the original matrix.
    np.linalg.cholesky(corr_matrix)
    # Give a clear, typed error path for mypy and callers.
    raise np.linalg.LinAlgError(
        f"Cholesky failed after {max_tries} tries (last jitter={jitter:g})."
    )


# ============================================================================
# Low-level sampler with a single global correlation
# ============================================================================
def sample_correlated_cluster(
    n_samples: int,
    n_features: int,
    rng: np.random.Generator,
    *,
    structure: CorrelationStructure,
    correlation: float | dict[int, float] | None = None,
) -> np.ndarray:
    """Sample a correlated cluster.

    Supports:
    - global mode: `correlation` is a float -> all samples share the same R.
    - class-specific mode: `correlation` is a dict -> samples are generated in
      contiguous blocks, one block per sorted mapping key. The total `n_samples`
      is split as evenly as possible across the keys (remainder to first blocks).

    Args:
        n_samples:
            Number of rows to generate.
        n_features:
            Number of columns in this cluster.
        rng:
            NumPy random Generator (use a shared one for reproducibility).
        structure:
            Correlation structure ("equicorrelated" or "toeplitz").
        correlation:
            Either a float (global mode) or a dict mapping class indices to
            correlation strengths (class-specific mode).
    Returns:
        Array X of shape (n_samples, n_features), dtype float64.
    Raises:
        TypeError: if `correlation` is None or a class-specific mapping with < 2 classes.
    """
    if correlation is None:
        raise TypeError("correlation must be provided (float for global or dict for class-specific).")

    # Helper to sample given rho for a number of samples
    def _sample_block(n_block: int, rho: float) -> np.ndarray:
        R = build_correlation_matrix(n_features, float(rho), structure)
        L = _cholesky_with_jitter(R, initial_jitter=1e-12, growth=10.0, max_tries=8)
        Z = rng.standard_normal(size=(n_block, n_features))
        return Z @ L.T

    # Class-specific mapping -> split n_samples into contiguous blocks (sorted keys)
    if isinstance(correlation, dict):
        if len(correlation) < 2:
            raise TypeError("class-specific correlation mapping must contain at least two classes.")
        keys = sorted(correlation.keys())
        k = len(keys)
        base = n_samples // k
        remainder = n_samples - base * k
        counts = [base + (1 if i < remainder else 0) for i in range(k)]

        parts: list[np.ndarray] = []
        for cls_key, cnt in zip(keys, counts):
            if cnt <= 0:
                # Avoid generating empty blocks; keep shape consistency by skipping
                parts.append(np.empty((0, n_features)))
                continue
            rho = float(correlation[cls_key])
            parts.append(_sample_block(cnt, rho))
        return np.vstack(parts)

    # Global mode (single float)
    assert isinstance(correlation, float)
    return _sample_block(n_samples, correlation)


# ============================================================================
# High-level sampler for all clusters (global + class-specific)
# ============================================================================
def sample_all_correlated_clusters(
    cfg: DatasetConfig,
    rng: np.random.Generator,
    y: np.ndarray,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Sample all correlated feature clusters as specified in cfg.

    This function supports both global and class-specific correlations
    as defined in CorrClusterConfig:

    - Global mode:
        CorrClusterConfig.correlation is a float.
        All samples share the same within-cluster correlation.

    - Class-specific mode:
        CorrClusterConfig.correlation is a dict {class_idx: rho}.
        Classes listed in the dict use their specified rho.
        Classes not listed use rho = 0.0 (independent cluster).

    Args:
        cfg:
            DatasetConfig with corr_clusters defined.
        rng:
            Shared NumPy random Generator.
        y:
            Class labels array of shape (n_samples,).

    Returns:
        x_clusters:
            Array of shape (n_samples, total_cluster_features) concatenated in
            the same order as cfg.corr_clusters.
        cluster_meta:
            Dict of per-cluster metadata:
            {
                "anchor_role":       {cluster_id: str, ...},
                "anchor_effect_size":{cluster_id: float | str | None, ...},
                "anchor_class":      {cluster_id: int | None, ...},
                "label":             {cluster_id: str | None, ...},
            }
    """
    n_samples = cfg.n_samples
    clusters: list[CorrClusterConfig] = cfg.corr_clusters or []
    cluster_cfgs: list[CorrClusterConfig] = []

    cluster_array_list: list[NDArray[np.float64]] = []
    for cluster_cfg in clusters:
        k = cluster_cfg.n_cluster_features

        if not cluster_cfg.is_class_specific():
            # Global correlation mode: one rho for all samples
            corr_global = float(cluster_cfg.correlation)  # type: ignore[arg-type]
            x_cluster = sample_correlated_cluster(
                n_samples=n_samples,
                n_features=k,
                rng=rng,
                structure=cluster_cfg.structure,
                correlation=corr_global,
            )
        else:
            # Class-specific mode: build block per class and stitch together
            x_cluster = np.empty((n_samples, k), dtype=np.float64)
            unique_classes = np.unique(y)

            for cls in unique_classes:
                class_mask = (y == cls)
                n_in_class = int(class_mask.sum())
                if n_in_class == 0:
                    continue

                corr_cls = cluster_cfg.get_correlation_for_class(int(cls))

                if abs(corr_cls) < 1e-15:
                    # Independent features for this class: standard normal noise
                    block = rng.standard_normal(
                        size=(n_in_class, k),
                        dtype=np.float64,
                    )
                else:
                    R_cls = build_correlation_matrix(
                        n_features=k,
                        correlation=corr_cls,
                        structure=cluster_cfg.structure,
                    )
                    L_cls = _cholesky_with_jitter(R_cls)
                    standard_normal_block = rng.standard_normal(
                        size=(n_in_class, k),
                        dtype=np.float64,
                    )
                    block = (standard_normal_block @ L_cls.T).astype(
                        np.float64,
                        copy=False,
                    )

                x_cluster[class_mask] = block

        cluster_array_list.append(x_cluster)
        cluster_cfgs.append(cluster_cfg)

    if cluster_cfgs:
        x_clusters = np.hstack(cluster_array_list)
    else:
        # No clusters configured → empty array with n_samples rows
        x_clusters = np.empty((y.shape[0], 0), dtype=float)

    # Aggregate meta fields into dicts keyed by cluster index (0-based)
    cluster_meta = {
        "anchor_role": {
            cluster_id: cfg.anchor_role for cluster_id, cfg in enumerate(cluster_cfgs)
        },
        "anchor_effect_size": {
            cluster_id: cfg.anchor_effect_size for cluster_id, cfg in enumerate(cluster_cfgs)
        },
        "anchor_class": {
            cluster_id: cfg.anchor_class for cluster_id, cfg in enumerate(cluster_cfgs)
        },
        "label": {
            cluster_id: cfg.label for cluster_id, cfg in enumerate(cluster_cfgs)
        },
    }

    return x_clusters, cluster_meta
