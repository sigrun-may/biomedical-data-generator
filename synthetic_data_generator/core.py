from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DatasetMeta:
    """Container for ground-truth metadata returned by `generate_dataset`."""
    feature_names: List[str]
    informative_idx: List[int]
    pseudo_idx: List[int]
    noise_idx: List[int]
    y_weights: Tuple[float, float]
    class_sep: float
    corr_within: float
    corr_between: float
    block_sizes: List[int]
    random_state: Optional[int]
    y_counts: Dict[int, int]


def _nearest_psd(corr: NDArray[np.float64], eps: float = 1e-8) -> NDArray[np.float64]:
    """Project a symmetric matrix to the nearest PSD by clipping eigenvalues.

    This allows slightly aggressive correlation settings to still work.
    """
    # Force symmetry
    sym = (corr + corr.T) / 2.0
    vals, vecs = np.linalg.eigh(sym)
    vals_clipped = np.clip(vals, eps, None)
    psd = (vecs * vals_clipped) @ vecs.T
    # Normalize diagonal to 1.0 (correlation matrix)
    d = np.sqrt(np.diag(psd))
    psd /= np.outer(d, d)
    np.fill_diagonal(psd, 1.0)
    return psd


def _make_block_corr(block_sizes: List[int], corr_within: float, corr_between: float) -> NDArray[np.float64]:
    """Construct a block correlation matrix with given within/between correlations."""
    p = int(sum(block_sizes))
    if p <= 0:
        raise ValueError("Sum of block_sizes must be > 0.")
    if not (-0.99 <= corr_between < 1.0):
        raise ValueError("corr_between must be in [-0.99, 1.0).")
    if not (-0.99 <= corr_within < 1.0):
        raise ValueError("corr_within must be in [-0.99, 1.0).")

    corr = np.full((p, p), corr_between, dtype=np.float64)
    start = 0
    for size in block_sizes:
        stop = start + size
        corr[start:stop, start:stop] = corr_within
        start = stop
    np.fill_diagonal(corr, 1.0)
    # Make sure it's PSD
    return _nearest_psd(corr)


def generate_dataset(
    n_samples: int = 300,
    n_features: int = 20,
    n_informative: int = 5,
    class_sep: float = 1.2,
    weights: Tuple[float, float] | None = None,   # class imbalance (p(y=0), p(y=1))
    random_state: int | None = 42,

    # Noise / irrelevant features
    n_noise: int = 0,
    noise_dist: str = "normal",                    # "normal" | "uniform"

    # Correlations (applied to ALL non-noise features)
    corr_matrix: NDArray[np.float64] | None = None,   # full corr matrix or None
    block_sizes: List[int] | None = None,             # alternative to corr_matrix
    corr_within: float = 0.8,
    corr_between: float = 0.0,

    # Pseudo-class confounding (independent of true y)
    n_pseudo: int = 0,
    pseudo_effect: float = 0.0,  # mean shift magnitude applied to pseudo features
) -> Tuple[NDArray[np.float64], NDArray[np.int64], DatasetMeta]:
    """
    Generate a high-dimensional synthetic binary classification dataset for
    biological/biomedical teaching and benchmarking.

    The generator supports:
      - Binary labels with optional class imbalance (`weights`)
      - Correlated feature blocks (or custom correlation matrix)
      - Informative features with class-dependent mean shift (`class_sep`)
      - Optional pseudo-class (confounder) that shifts a subset of features but
        is independent of the true label (useful to demonstrate false positives)
      - Pure noise features drawn independent from the correlated structure

    Parameters
    ----------
    n_samples : int, default=300
        Number of samples.
    n_features : int, default=20
        Total number of features = informative + pseudo + (correlated-but-uninformative) + noise.
    n_informative : int, default=5
        Number of informative features (class-separating).
    class_sep : float, default=1.2
        Target mean difference (in SD units) between the two classes on informative features.
        Implemented as +/- class_sep/2 shift for y=1 / y=0 respectively.
    weights : (float, float) or None, default=None
        Class probabilities (p(y=0), p(y=1)). If None, classes are balanced (0.5, 0.5).
    random_state : int or None, default=42
        Seed for reproducibility.

    n_noise : int, default=0
        Number of fully independent noise features (NOT part of the correlation structure).
    noise_dist : {"normal", "uniform"}, default="normal"
        Distribution for pure noise features.

    corr_matrix : ndarray of shape (p_corr, p_corr) or None, default=None
        Full correlation matrix for the non-noise features (p_corr = n_features - n_noise).
        Must be symmetric with ones on the diagonal; will be projected to PSD if needed.
    block_sizes : list[int] or None, default=None
        If `corr_matrix` is None, build a block correlation matrix with these block sizes.
        If None, a single block of size p_corr is used.
    corr_within : float, default=0.8
        Within-block correlation used when `block_sizes` is provided.
    corr_between : float, default=0.0
        Between-block correlation used when `block_sizes` is provided.

    n_pseudo : int, default=0
        Number of pseudo-class features (confounded but NOT informative for y).
        These are part of the correlated set if `n_noise < n_features`.
    pseudo_effect : float, default=0.0
        Mean shift magnitude applied to pseudo features based on an *independent* binary
        pseudo label z ~ Bernoulli(0.5). Shift is +/- pseudo_effect/2.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Binary labels {0, 1}.
    meta : DatasetMeta
        Ground-truth metadata (feature roles, block info, label distribution, etc.).

    Notes
    -----
    - Correlation is applied to all *non-noise* features. Pure noise features are appended
      on the right and are independent of everything else.
    - Informative features receive class-dependent mean shifts; pseudo features receive
      pseudo-class-dependent shifts independent of y.
    - The returned `meta` contains index lists *after* the final column order.

    Examples
    --------
    >>> X, y, meta = generate_dataset(
    ...     n_samples=120, n_features=200, n_informative=10,
    ...     class_sep=1.0, n_noise=190, corr_within=0.75, random_state=42
    ... )
    >>> X.shape, y.shape
    ((120, 200), (120,))
    """
    if n_samples <= 1:
        raise ValueError("n_samples must be > 1.")
    if n_features <= 0:
        raise ValueError("n_features must be > 0.")
    if n_informative < 0:
        raise ValueError("n_informative must be >= 0.")
    if n_noise < 0:
        raise ValueError("n_noise must be >= 0.")
    if n_pseudo < 0:
        raise ValueError("n_pseudo must be >= 0.")
    if n_informative + n_pseudo > n_features:
        raise ValueError("n_informative + n_pseudo cannot exceed n_features.")
    if n_noise > n_features:
        raise ValueError("n_noise cannot exceed n_features.")

    # Effective counts
    p_total = int(n_features)
    p_noise = int(n_noise)
    p_corr = p_total - p_noise  # features under the correlation structure

    if n_informative > p_corr:
        raise ValueError("n_informative cannot exceed (n_features - n_noise).")
    if n_pseudo > p_corr - n_informative:
        raise ValueError("n_pseudo cannot exceed (n_features - n_noise - n_informative).")

    # Remaining correlated-but-uninformative features (not pseudo)
    p_corr_plain = p_corr - n_informative - n_pseudo

    # RNG
    rng = np.random.default_rng(random_state)

    # Labels with optional class imbalance
    if weights is None:
        w0, w1 = 0.5, 0.5
    else:
        if not (len(weights) == 2 and math.isfinite(weights[0]) and math.isfinite(weights[1])):
            raise ValueError("weights must be a tuple (p0, p1).")
        w0, w1 = float(weights[0]), float(weights[1])
        if w0 < 0 or w1 < 0 or abs((w0 + w1) - 1.0) > 1e-8:
            raise ValueError("weights must be non-negative and sum to 1.")
    y = rng.choice(np.array([0, 1], dtype=np.int64), size=n_samples, p=[w0, w1]).astype(np.int64)

    # Pseudo-class (independent of true y)
    z = rng.integers(0, 2, size=n_samples).astype(np.int64) if n_pseudo > 0 else None

    # Build correlation for the correlated part (p_corr may be 0)
    X_corr = np.empty((n_samples, 0), dtype=np.float64)
    block_used: List[int] = []

    if p_corr > 0:
        if corr_matrix is not None:
            corr = np.asarray(corr_matrix, dtype=np.float64)
            if corr.shape != (p_corr, p_corr):
                raise ValueError(f"corr_matrix must be shape ({p_corr}, {p_corr}).")
            if not np.allclose(np.diag(corr), 1.0, atol=1e-6):
                raise ValueError("corr_matrix must have ones on the diagonal.")
            corr = _nearest_psd(corr)
            block_used = [p_corr]  # unknown block layout; treat as single
        else:
            if block_sizes is None or sum(block_sizes) != p_corr:
                # Default: single block covering p_corr
                block_used = [p_corr] if block_sizes is None else block_sizes + [p_corr - sum(block_sizes)]
            else:
                block_used = block_sizes
            corr = _make_block_corr(block_used, corr_within=corr_within, corr_between=corr_between)

        # Sample correlated base: mean 0, diag 1
        X_corr = rng.multivariate_normal(mean=np.zeros(p_corr), cov=corr, size=n_samples, check_valid="ignore")
        X_corr = X_corr.astype(np.float64, copy=False)

        # Indices within correlated part
        idx_inf = np.arange(0, n_informative, dtype=np.int64)
        idx_pseudo = np.arange(n_informative, n_informative + n_pseudo, dtype=np.int64)
        idx_plain = np.arange(n_informative + n_pseudo, p_corr, dtype=np.int64)

        # Class separation for informative features
        if n_informative > 0 and class_sep != 0.0:
            # Shift magnitude: +/- class_sep/2 to produce total mean difference ≈ class_sep
            shift = (class_sep / 2.0).astype(np.float64) if isinstance(class_sep, np.ndarray) else float(class_sep) / 2.0
            # Broadcast shifts per sample
            s = np.where(y == 1, +shift, -shift).reshape(-1, 1)
            X_corr[:, idx_inf] = X_corr[:, idx_inf] + s

        # Pseudo-class shifts (independent confounding)
        if n_pseudo > 0 and pseudo_effect != 0.0 and z is not None:
            pshift = (float(pseudo_effect) / 2.0)
            s_p = np.where(z == 1, +pshift, -pshift).reshape(-1, 1)
            X_corr[:, idx_pseudo] = X_corr[:, idx_pseudo] + s_p

    # Pure noise features (independent of everything)
    X_noise = np.empty((n_samples, 0), dtype=np.float64)
    if p_noise > 0:
        if noise_dist not in {"normal", "uniform"}:
            raise ValueError("noise_dist must be 'normal' or 'uniform'.")
        if noise_dist == "normal":
            X_noise = rng.normal(loc=0.0, scale=1.0, size=(n_samples, p_noise)).astype(np.float64)
        else:
            # Uniform in [-1, 1] roughly matches variance scale of N(0,1)
            X_noise = rng.uniform(low=-1.0, high=1.0, size=(n_samples, p_noise)).astype(np.float64)

    # Concatenate correlated part and noise part
    X = np.hstack([X_corr, X_noise]) if p_noise > 0 else X_corr

    # Build final indexing (correlated part comes first, then noise)
    informative_idx = list(range(0, n_informative))
    pseudo_idx = list(range(n_informative, n_informative + n_pseudo))
    noise_idx = list(range(p_corr, p_corr + p_noise))

    # Feature names
    feature_names = [
        *(f"inf_{i}" for i in range(n_informative)),
        *(f"pseudo_{i}" for i in range(n_pseudo)),
        *(f"feat_{i}" for i in range(p_corr_plain)),
        *(f"noise_{i}" for i in range(p_noise)),
    ]

    meta = DatasetMeta(
        feature_names=list(feature_names),
        informative_idx=informative_idx,
        pseudo_idx=pseudo_idx,
        noise_idx=noise_idx,
        y_weights=(w0, w1),
        class_sep=float(class_sep),
        corr_within=float(corr_within),
        corr_between=float(corr_between),
        block_sizes=list(block_used) if block_used else [],
        random_state=random_state,
        y_counts={int(c): int((y == c).sum()) for c in (0, 1)},
    )

    return X, y, meta
