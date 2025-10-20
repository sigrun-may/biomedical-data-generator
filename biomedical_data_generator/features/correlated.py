# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Functions to generate correlated feature clusters simulating biomarker patterns."""
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from ..config import CorrCluster


# =========================
# Covariance matrix builders
# =========================
def _cov_equicorr(n_cluster_features: int, rho: float) -> NDArray[np.float64]:
    """Build equicorrelated covariance matrix.

    All pairs have the same correlation rho.
    Typical for tightly regulated biological pathways.

    Args:
        n_cluster_features: Number of markers in the cluster.
        rho: Correlation strength (0 ≤ rho < 1).

    Returns:
        Covariance matrix of shape (n_cluster_features, n_cluster_features).
    """
    identity: NDArray[np.float64] = np.eye(n_cluster_features, dtype=np.float64)
    ones: NDArray[np.float64] = np.ones((n_cluster_features, n_cluster_features), dtype=np.float64)
    return (1 - rho) * identity + rho * ones


def _cov_toeplitz(n_cluster_features: int, rho: float) -> NDArray[np.float64]:
    """Build Toeplitz (AR-1 like) covariance matrix.

    Correlation decays with distance: rho**|i-j|.
    Useful for ordered biomarkers (e.g., time series, spatial gradients).

    Args:
        n_cluster_features: Number of markers in the cluster.
        rho: Base correlation (-1 < rho < 1).

    Returns:
        Covariance matrix of shape (n_cluster_features, n_cluster_features).
    """
    idx = np.arange(n_cluster_features, dtype=np.int64)
    D: NDArray[np.float64] = np.abs(idx[:, None] - idx[None, :]).astype(np.float64, copy=False)
    return np.asarray(rho**D, dtype=np.float64)


def sample_cluster_matrix(n: int, cluster: CorrCluster, rng: np.random.Generator) -> NDArray[np.float64]:
    """Sample a correlated biomarker cluster from a multivariate normal distribution.

    Args:
        n: Number of samples (patients).
        cluster: Cluster configuration with correlation structure.
        rng: Random number generator for reproducibility.

    Returns:
        Feature matrix of shape (n, cluster.n_cluster_features) with standardized columns.
        Columns represent correlated biomarkers.

    Raises:
        ValueError: If cluster parameters are invalid.

    Examples:
    --------
        >>> from biomedical_data_generator import CorrCluster
        >>> import numpy as np
        >>>
        >>> inflammation = CorrCluster(n_cluster_features=5, rho=0.8, label="Cytokines")
        >>> rng = np.random.default_rng(42)
        >>> X = sample_cluster_matrix(n=200, cluster=inflammation, rng=rng)
        >>> X.shape
        (200, 5)
        >>> np.corrcoef(X, rowvar=False).mean()  # should be close to 0.8
        0.79...
    """
    # Build covariance matrix based on structure
    if cluster.structure == "equicorrelated":
        Sigma = _cov_equicorr(cluster.n_cluster_features, cluster.rho)
    else:  # toeplitz
        Sigma = _cov_toeplitz(cluster.n_cluster_features, cluster.rho)

    # Cholesky decomposition for efficient sampling
    L = np.linalg.cholesky(Sigma)

    # Sample from standard normal and transform
    Z: NDArray[np.float64] = rng.normal(size=(n, cluster.n_cluster_features)).astype(np.float64, copy=False)
    X: NDArray[np.float64] = cast(NDArray[np.float64], Z @ L.T)

    # Standardize columns to unit variance (helpful for consistent effect sizes)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    return X


# ============================================
# Public API: Standalone cluster generation
# ============================================
# NOTE: These functions are primarily for educational/exploratory purposes.
# For production use, prefer using CorrCluster within generate_dataset().


def generate_correlated_cluster(
    n_samples: int,
    n_cluster_features: int,
    rho: float = 0.7,
    structure: Literal["equicorrelated", "toeplitz"] = "equicorrelated",
    rng: np.random.Generator | None = None,
    label: str | None = None,
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Generate a single correlated biomarker cluster without class labels.

    **Educational Use**:  Generate a single correlated feature cluster (no labels y involved). Explore correlation
    patterns in isolation before integrating into a full dataset. For production datasets, use CorrCluster
    with generate_dataset() instead.

    Returns (X_cluster, meta) where meta contains the empirical correlation matrix.

    Args:
        n_samples: Number of samples (patients/observations).
        n_cluster_features: Number of biomarkers in the cluster.
        rho: Target correlation strength.
            - 0.0 = independent
            - 0.5 = moderate correlation
            - 0.8+ = strong correlation (pathway-like)
        structure: Correlation pattern.
            - "equicorrelated": constant pairwise correlation
            - "toeplitz": correlation decays with distance
        rng: Random number generator (if None, creates a new one).
        label: Optional descriptive label (e.g., "Inflammation panel").

    Returns:
        tuple:
            - X (np.ndarray): Shape (n_samples, n_cluster_features) with standardized columns.
            - meta (dict): Metadata containing:
                * n_cluster_features: cluster size
                * rho: target correlation
                * structure: correlation pattern
                * label: descriptive label
                * corr_matrix: empirical correlation matrix (n_cluster_features × n_cluster_features)
                * mean_offdiag: mean of off-diagonal correlations
                * min_offdiag: minimum off-diagonal correlation

    Raises:
        ValueError: If n_cluster_features < 1 or rho not in [0, 1).

    Examples:
    --------
        Generate a strong inflammatory marker cluster:

        >>> X, meta = generate_correlated_cluster(
        ...     n_samples=200,
        ...     n_cluster_features=5,
        ...     rho=0.85,
        ...     label="Acute phase proteins"
        ... )
        >>> X.shape
        (200, 5)
        >>> meta['mean_offdiag']  # actual correlation achieved
        0.84...

        Weak age-related confounders:

        >>> X, meta = generate_correlated_cluster(
        ...     n_samples=150,
        ...     n_cluster_features=3,
        ...     rho=0.4,
        ...     label="Age effects"
        ... )

    Notes:
    -----
        Medical interpretation:
        - rho=0.85: Very strong (tightly regulated pathway)
        - rho=0.60: Strong (coordinated response)
        - rho=0.40: Moderate (loose biological coupling)
        - rho=0.20: Weak (barely related markers)

        The empirical correlation may deviate slightly from rho due to
        finite sample size. Use find_seed_for_correlation() for precise control.

        For production use, prefer using CorrCluster within generate_dataset().
    """
    # Validation
    if n_cluster_features < 1:
        raise ValueError("n_cluster_features must be >= 1")

    if structure == "equicorrelated":
        if not (0.0 <= rho < 1.0):
            raise ValueError(
                f"For equicorrelated structure, rho must be in [0, 1), got {rho}. "
                f"Hint: 0=independent, 0.5=moderate, 0.8=strong"
            )
    else:  # toeplitz
        if not (-0.999 < rho < 0.999):
            raise ValueError(
                f"For toeplitz structure, |rho| must be < 1, got {rho}. "
                f"Note: Negative rho creates alternating correlations."
            )

    # Initialize RNG if not provided
    rng = np.random.default_rng() if rng is None else rng

    # Choose structure for the target covariance
    cov: NDArray[np.float64] = (
        _cov_equicorr(n_cluster_features, rho)
        if structure == "equicorrelated"
        else _cov_toeplitz(n_cluster_features, rho)
    )

    # Cholesky factor: cov = chol_lower @ chol_lower.T  (lower-triangular)
    chol_lower: NDArray[np.float64] = np.linalg.cholesky(cov)

    # Standard-normal noise: shape = (n_samples, n_cluster_features)
    noise_std: NDArray[np.float64] = rng.normal(size=(n_samples, n_cluster_features)).astype(np.float64, copy=False)

    # Impose the target covariance: X has Cov ≈ cov
    # (matrix multiply with chol_lower.T)
    X: NDArray[np.float64] = cast(NDArray[np.float64], noise_std @ chol_lower.T)

    # Standardize to unit variance
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    # Compute empirical correlation
    C: NDArray[np.float64] = np.asarray(np.corrcoef(X, rowvar=False), dtype=np.float64)

    # Size-1 edge case: np.corrcoef may return a 0-D scalar -> promote to (1, 1)
    if C.ndim == 0:
        C = C.reshape(1, 1)

    # Off-diagonal metrics
    if n_cluster_features > 1:
        diag_mask = np.eye(n_cluster_features, dtype=bool)  # boolean diagonal mask (NumPy 2.0 compatible)
        off_diag = C[~diag_mask]
        mean_offdiag = float(off_diag.mean()) if off_diag.size else 1.0
        min_offdiag = float(off_diag.min()) if off_diag.size else 1.0
    else:
        mean_offdiag = 1.0
        min_offdiag = 1.0

    meta: dict[str, object] = {
        "n_cluster_features": int(n_cluster_features),
        "rho": float(rho),
        "structure": structure,
        "label": label,
        "corr_matrix": C,  # empirical correlation (n_cluster_features x n_cluster_features)
        "mean_offdiag": mean_offdiag,
        "min_offdiag": min_offdiag,
    }
    return X, meta


# =====================================================
# Public: search a seed until correlation is sufficient
# =====================================================
def find_seed_for_correlation(
    n_samples: int,
    n_cluster_features: int,
    rho_target: float,
    structure: Literal["equicorrelated", "toeplitz"] = "equicorrelated",
    metric: Literal["mean_offdiag", "min_offdiag"] = "mean_offdiag",
    threshold: float = 0.65,
    op: Literal[">=", "<="] = ">=",
    tol: float | None = 0.02,
    start_seed: int = 0,
    max_tries: int = 500,
) -> tuple[int, dict[str, object]]:
    """Search for a random seed that achieves target correlation quality.

        Tries multiple seeds until finding one where the empirical correlation
        matches the target within tolerance. Useful where precise correlation is needed.

        Try seeds starting from `start_seed` until one of the following is satisfied:
      - |mean_offdiag - rho_target| <= tol (if tol is not None), else
      - (metric op threshold) with metric in {"mean_offdiag", "min_offdiag"} and op in {">=", "<="}.

    Args:
        n_samples: Number of samples (rows).
        n_cluster_features: Number of features (columns) within cluster.
        rho_target: Desired correlation strength.
        structure: "equicorrelated" or "toeplitz".
        metric: Quality metric to optimize.
            - "mean_offdiag": average off-diagonal correlation
            - "min_offdiag": minimum off-diagonal correlation
        threshold: Minimum acceptable value for metric (if tol is None).
        op: Comparison operator for threshold (">=", "<=").
        tol: If provided, accept when |mean_offdiag - rho_target| <= tol.
            This takes precedence over threshold.
        start_seed: First seed to try.
        max_tries: Maximum number of seeds to try before giving up.

    Returns:
        tuple:
            seed (int): The first seed that satisfied the condition.
            meta (dict): Metadata as returned by generate_correlated_cluster.

    Raises:
        RuntimeError: If no seed satisfied the rule within max_tries.
        ValueError: If n_cluster_features < 1 or rho_target not in [0, 1).

    Examples:
    --------
        Find seed for very strong inflammation markers:

        >>> seed, meta = find_seed_for_correlation(
        ...     n_samples=200,
        ...     n_cluster_features=6,
        ...     rho_target=0.85,
        ...     tol=0.03
        ... )
        >>> meta['mean_offdiag']
        0.847...

        Ensure minimum correlation for teaching:

        >>> seed, meta = find_seed_for_correlation(
        ...     n_samples=150,
        ...     n_cluster_features=4,
        ...     rho_target=0.70,
        ...     metric="min_offdiag",
        ...     threshold=0.65,
        ...     tol=None
        ... )

    Notes:
    -----
        Medical interpretation:
        - Use this when demonstrating correlation patterns in teaching
        - Higher n_cluster_features and rho_target may require more tries
        - For large clusters (n_cluster_features > 10), consider increasing max_tries

        For tol-based acceptance we require n_cluster_features <= n_samples.
        When p >> n, the mean off-diagonal can appear artificially close to rho_target
        due to heavy averaging; in such cases, prefer threshold mode or increase n.
    """
    # Validation
    if n_cluster_features < 1:
        raise ValueError("n_cluster_features must be >= 1")

    if structure == "equicorrelated":
        if not (0.0 <= rho_target < 1.0):
            raise ValueError(f"For equicorrelated, rho_target must be in [0, 1), got {rho_target}")
    else:
        if not (-0.999 < rho_target < 0.999):
            raise ValueError(f"For toeplitz, |rho_target| must be < 1, got {rho_target}")

    seed = start_seed
    for _ in range(max_tries):
        rng = np.random.default_rng(seed)
        _, meta_data = generate_correlated_cluster(n_samples, n_cluster_features, rho_target, structure, rng=rng)
        mean_off = cast(float, meta_data["mean_offdiag"])
        min_off = cast(float, meta_data["min_offdiag"])

        # Acceptance rule:
        # If tol is provided -> ONLY use tolerance (ignore threshold).
        # Guard: require p <= n for tol-based acceptance to avoid p>>n averaging artifacts.
        # If tol is None     -> fall back to (metric op threshold).
        if tol is not None:
            # High-dimensional guard
            if n_cluster_features <= n_samples and abs(mean_off - rho_target) <= tol:
                return seed, meta_data
        else:
            metric_value = mean_off if metric == "mean_offdiag" else min_off
            if (op == ">=" and metric_value >= threshold) or (op == "<=" and metric_value <= threshold):
                return seed, meta_data
        seed += 1

    # Failed to find suitable seed
    raise RuntimeError(
        f"Failed to find seed satisfying correlation criterion within {max_tries} tries. "
        f"Target: rho={rho_target}, tol={tol}, metric={metric}, threshold={threshold}. "
        f"Consider: (1) increasing max_tries, (2) relaxing tol/threshold, "
        f"(3) reducing n_cluster_features or rho_target."
    )
