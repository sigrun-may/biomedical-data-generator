# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Functions to generate correlated feature clusters."""

from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from ..config import CorrCluster


# =========================
# Small covariance helpers
# =========================
def _cov_equicorr(size: int, rho: float) -> NDArray[np.float64]:
    identity: NDArray[np.float64] = np.eye(size, dtype=np.float64)
    ones: NDArray[np.float64] = np.ones((size, size), dtype=np.float64)
    return (1 - rho) * identity + rho * ones


def _cov_toeplitz(size: int, rho: float) -> NDArray[np.float64]:
    idx = np.arange(size, dtype=np.int64)
    D: NDArray[np.float64] = np.abs(idx[:, None] - idx[None, :]).astype(np.float64, copy=False)
    # ensure float64 ndarray for typing
    return np.asarray(rho**D, dtype=np.float64)


def sample_cluster_matrix(n: int, cluster: CorrCluster, rng: np.random.Generator) -> NDArray[np.float64]:
    """Sample a feature matrix X for a single correlated cluster.

    Args:
        n: Number of samples (rows).
        cluster: CorrCluster configuration.
        rng: Random number generator.

    Returns:
        X (np.ndarray): Shape (n, cluster.size) with standardized columns.

    Raises:
        ValueError: If cluster.size < 1 or cluster.rho not in valid range.
    """
    Sigma = (
        _cov_equicorr(cluster.size, cluster.rho)
        if cluster.structure == "equicorrelated"
        else _cov_toeplitz(cluster.size, cluster.rho)
    )
    L = np.linalg.cholesky(Sigma)
    Z: NDArray[np.float64] = rng.normal(size=(n, cluster.size)).astype(np.float64, copy=False)
    X: NDArray[np.float64] = cast(NDArray[np.float64], Z @ L.T)
    # standardize columns to ~unit variance (helpful for teaching consistency)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    return X  # (n, size)


# ============================================
# Public: generate a single correlated cluster
# ============================================
def generate_correlated_cluster(
    n_samples: int,
    size: int,
    rho: float = 0.7,
    structure: Literal["equicorrelated", "toeplitz"] = "equicorrelated",
    rng: np.random.Generator | None = None,
    label: str | None = None,
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Generate a single correlated feature cluster (no labels y involved).

    Returns (X_cluster, meta) where meta contains the empirical correlation matrix.

    Args:
        n_samples: Number of samples (rows).
        size: Number of features (columns).
        rho: Target correlation between features (0 ≤ rho < 1).
        structure: "equicorrelated" or "toeplitz".
        rng: Optional random number generator (if None, a new one is created).
        label: Optional didactic tag for this cluster.

    Returns:
        tuple:
            X (np.ndarray): Shape (n_samples, size) with standardized columns.
            meta (dict): Metadata with keys:
              size, rho, structure, label, corr_matrix (size x size),
              mean_offdiag, min_offdiag.

    Raises:
        ValueError: If size < 1 or rho not in [0, 1).
    """
    if size < 1:
        raise ValueError("size must be >= 1")

    if structure == "equicorrelated":
        if not (0.0 <= rho < 1.0):
            raise ValueError("for equicorrelated: rho must be in [0, 1)")
    else:  # toeplitz (AR(1)-artige Struktur)
        if not (-0.999 < rho < 0.999):
            raise ValueError("for toeplitz: |rho| must be < 1")

    rng = np.random.default_rng() if rng is None else rng
    Sigma: NDArray[np.float64] = _cov_equicorr(size, rho) if structure == "equicorrelated" else _cov_toeplitz(size, rho)
    L = np.linalg.cholesky(Sigma)
    Z: NDArray[np.float64] = rng.normal(size=(n_samples, size)).astype(np.float64, copy=False)
    X: NDArray[np.float64] = cast(NDArray[np.float64], Z @ L.T)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    C: NDArray[np.float64] = np.asarray(np.corrcoef(X, rowvar=False), dtype=np.float64)

    # boolean diagonal mask (NumPy 2.0 compatible)
    diag_mask = np.eye(size, dtype=bool)
    off_diag = C[~diag_mask]

    meta: dict[str, object] = {
        "size": int(size),
        "rho": float(rho),
        "structure": structure,
        "label": label,
        "corr_matrix": C,  # empirical correlation (size x size)
        "mean_offdiag": float(off_diag.mean()) if size > 1 else 1.0,
        "min_offdiag": float(off_diag.min()) if size > 1 else 1.0,
    }
    return X, meta


# =====================================================
# Public: search a seed until correlation is sufficient
# =====================================================
def find_seed_for_correlation(
    n_samples: int,
    size: int,
    rho_target: float,
    structure: Literal["equicorrelated", "toeplitz"] = "equicorrelated",
    metric: Literal["mean_offdiag", "min_offdiag"] = "mean_offdiag",
    threshold: float = 0.65,
    op: Literal[">=", "<="] = ">=",
    tol: float | None = 0.02,
    start_seed: int = 0,
    max_tries: int = 500,
) -> tuple[int, dict[str, object]]:
    """Try seeds until the empirical correlation satisfies the rule.

    Try seeds starting from `start_seed` until one of the following is satisfied:
      - |mean_offdiag - rho_target| <= tol (if tol is not None), else
      - (metric op threshold) with metric in {"mean_offdiag", "min_offdiag"} and op in {">=", "<="}.

    Args:
        n_samples: Number of samples (rows).
        size: Number of features (columns).
        rho_target: Target correlation between features.
        structure: "equicorrelated" or "toeplitz".
        metric: Empirical metric to use for acceptance.
        threshold: Threshold for the metric (if tol is None).
        op: Operator for threshold comparison ("<=" or ">=").
        tol: Optional tolerance around rho_target for acceptance.
        start_seed: First seed to try.
        max_tries: Maximum number of seeds to try before giving up.

    Returns:
        tuple:
            seed (int): The first seed that satisfied the condition.
            meta (dict): Metadata as returned by generate_correlated_cluster.

    Raises:
        RuntimeError: If no seed satisfied the rule within max_tries.
        ValueError: If size < 1 or rho_target not in [0, 1).
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    if not (0.0 <= rho_target < 1.0):
        raise ValueError("rho_target must be in [0, 1)")

    seed = start_seed
    for _ in range(max_tries):
        rng = np.random.default_rng(seed)
        _, meta_data = generate_correlated_cluster(n_samples, size, rho_target, structure, rng=rng)
        mean_off = cast(float, meta_data["mean_offdiag"])
        min_off = cast(float, meta_data["min_offdiag"])

        ok = False
        if tol is not None:
            ok = abs(mean_off - rho_target) <= tol
        if not ok:
            val = mean_off if metric == "mean_offdiag" else min_off
            ok = (val >= threshold) if op == ">=" else (val <= threshold)

        if ok:
            return seed, meta_data
        seed += 1
    raise RuntimeError("No seed satisfied the correlation rule within max_tries.")
