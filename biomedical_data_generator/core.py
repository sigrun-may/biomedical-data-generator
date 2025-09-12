# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

from __future__ import annotations

import math
import warnings as _warnings
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

NoiseDist = Literal["normal", "uniform", "lognormal"]


# ----- Ground-truth meta returned to the user ----------------------------------------------------
@dataclass(frozen=True)
class DatasetMeta:
    """Container for ground-truth metadata returned by `generate_dataset`."""

    feature_names: list[str]
    informative_idx: list[int]
    pseudo_idx: list[int]
    noise_idx: list[int]
    y_weights: tuple[float, float]
    class_sep: float
    corr_within: float
    corr_between: float
    block_sizes: list[int]
    random_state: int | None
    y_counts: dict[int, int]


# ----- Optional config holder (handy for presets / CLI), strictly additive -----------------------
@dataclass(frozen=True)
class DatasetConfig:
    # Mirrors the public function parameters (non-breaking convenience)
    n_samples: int = 300
    n_features: int = 20
    n_informative: int = 5
    class_sep: float = 1.2
    classes: int = 2  # only binary supported; kept here for completeness
    weights: tuple[float, float] | None = None
    random_state: int | None = 42

    # Noise / irrelevant features
    n_noise: int = 0
    noise_dist: NoiseDist = "normal"

    # Correlations
    corr_matrix: NDArray[np.float64] | None = None
    block_sizes: Iterable[int] | None = None
    corr_within: float = 0.8
    corr_between: float = 0.0

    # Pseudo-class (confounding)
    n_pseudo: int = 0
    pseudo_effect: float = 0.0

    # Deprecated convenience flags kept for compatibility (ignored by the core impl)
    add_pseudo: bool | None = None
    return_meta: bool = True  # handled at wrapper level only

    def to_kwargs(self) -> dict[str, Any]:
        """Convert dataclass to a kwargs dict suitable for `generate_dataset`."""
        d = asdict(self)
        # Normalize iterables to lists for downstream use
        if d.get("block_sizes") is not None:
            d["block_sizes"] = list(d["block_sizes"])  # type: ignore[index]
        return d


# ----- Public entry point ------------------------------------------------------------------------
def generate_dataset(
    n_samples: int = 300,
    n_features: int = 20,
    n_informative: int = 5,
    class_sep: float = 1.2,
    *,
    classes: int = 2,
    weights: tuple[float, float] | None = None,
    random_state: int | None = 42,
    # Noise / irrelevant features
    n_noise: int = 0,
    noise_dist: NoiseDist = "normal",
    # Correlations
    corr_matrix: NDArray[np.float64] | None = None,
    block_sizes: Iterable[int] | None = None,
    corr_within: float = 0.8,
    corr_between: float = 0.0,
    # Pseudo confounding
    n_pseudo: int = 0,
    pseudo_effect: float = 0.0,
    # Legacy convenience flags (wrapper-level only)
    add_pseudo: bool | None = None,   # deprecated: use pseudo_effect instead
    return_meta: bool = True,            # wrapper decides whether to drop meta
    # Optional config (base values) – explicit function args take precedence
    config: DatasetConfig | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64], DatasetMeta] | tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Stable public entry point used in README & docs.

    Notes on precedence
    -------------------
    If `config` is provided, it serves as a base. Explicit function arguments
    are applied on top (explicit args take precedence).

    Binary-only
    -----------
    The current implementation supports binary classification only (`classes=2`).

    Returns
    -------
    (X, y, meta) if return_meta=True, otherwise (X, y).
    """
    # --- Merge config (if any) with explicit parameters (explicit wins) --------------------------
    base: dict[str, Any] = config.to_kwargs() if config is not None else {}

    # Map legacy flag 'add_pseudo' to a default pseudo effect if user set it
    if add_pseudo:
        # If user did not set pseudo_effect explicitly, use a reasonable shift
        if pseudo_effect == 0.0:
            pseudo_effect = float(class_sep)

    # Enforce binary-only for now
    if classes != 2:
        raise NotImplementedError("Only binary classification (classes=2) is supported at the moment.")

    # Normalize/validate weights (must be two non-negative numbers summing to 1)
    norm_weights: tuple[float, float] | None
    if weights is None:
        norm_weights = None
    else:
        if len(weights) != 2:
            raise ValueError("weights must be a 2-tuple (p(y=0), p(y=1)).")
        w0, w1 = float(weights[0]), float(weights[1])
        if w0 < 0 or w1 < 0 or abs((w0 + w1) - 1.0) > 1e-8:
            raise ValueError("weights must be non-negative and sum to 1.")
        norm_weights = (w0, w1)

    # Normalize block_sizes to list for the core implementation
    block_list = list(block_sizes) if block_sizes is not None else None

    # Build final kwargs for the internal implementation
    impl_kwargs: dict[str, Any] = {
        # core parameters
        "n_samples": n_samples if n_samples is not None else base.get("n_samples"),
        "n_features": n_features if n_features is not None else base.get("n_features"),
        "n_informative": n_informative if n_informative is not None else base.get("n_informative"),
        "class_sep": class_sep if class_sep is not None else base.get("class_sep"),
        "weights": norm_weights if norm_weights is not None else base.get("weights"),
        "random_state": random_state if random_state is not None else base.get("random_state"),
        # noise
        "n_noise": n_noise if n_noise is not None else base.get("n_noise"),
        "noise_dist": noise_dist if noise_dist is not None else base.get("noise_dist"),
        # correlations
        "corr_matrix": corr_matrix if corr_matrix is not None else base.get("corr_matrix"),
        "block_sizes": block_list if block_list is not None else base.get("block_sizes"),
        "corr_within": corr_within if corr_within is not None else base.get("corr_within"),
        "corr_between": corr_between if corr_between is not None else base.get("corr_between"),
        # pseudo/confounding
        "n_pseudo": n_pseudo if n_pseudo is not None else base.get("n_pseudo"),
        "pseudo_effect": pseudo_effect if pseudo_effect is not None else base.get("pseudo_effect"),
    }

    # Warn about deprecated args that do nothing at core level
    if add_pseudo is not None:
        _warnings.warn(
            "Argument 'add_pseudo' is deprecated. Use 'n_pseudo' and 'pseudo_effect' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Call internal generator
    X, y, meta = _generate_dataset_impl(**impl_kwargs)

    # Respect return_meta flag at the wrapper level
    if not return_meta:
        return X, y
    return X, y, meta


# ----- Internals --------------------------------------------------------------------------------
def _nearest_psd(corr: NDArray[np.float64], eps: float = 1e-8) -> NDArray[np.float64]:
    """Project a symmetric matrix to the nearest PSD by clipping eigenvalues."""
    sym = (corr + corr.T) / 2.0
    vals, vecs = np.linalg.eigh(sym)
    vals_clipped = np.clip(vals, eps, None)
    psd = (vecs * vals_clipped) @ vecs.T
    # Normalize diagonal to 1.0 (correlation matrix)
    d = np.sqrt(np.diag(psd))
    psd /= np.outer(d, d)
    np.fill_diagonal(psd, 1.0)
    return psd


def _make_block_corr(block_sizes: list[int], corr_within: float, corr_between: float) -> NDArray[np.float64]:
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
    return _nearest_psd(corr)


def _generate_dataset_impl(
    n_samples: int = 300,
    n_features: int = 20,
    n_informative: int = 5,
    class_sep: float = 1.2,
    weights: tuple[float, float] | None = None,  # class imbalance (p(y=0), p(y=1))
    random_state: int | None = 42,
    # Noise / irrelevant features
    n_noise: int = 0,
    noise_dist: str = "normal",  # "normal" | "uniform" | "lognormal"
    # Correlations (applied to ALL non-noise features)
    corr_matrix: NDArray[np.float64] | None = None,  # full corr matrix or None
    block_sizes: list[int] | None = None,  # alternative to corr_matrix
    corr_within: float = 0.8,
    corr_between: float = 0.0,
    # Pseudo-class confounding (independent of true y)
    n_pseudo: int = 0,
    pseudo_effect: float = 0.0,  # mean shift magnitude applied to pseudo features
) -> tuple[NDArray[np.float64], NDArray[np.int64], DatasetMeta]:
    """
    Generate a high-dimensional synthetic *binary* classification dataset.

    Supports:
      - Class imbalance via `weights`
      - Correlated feature blocks or a custom correlation matrix
      - Informative features with class-dependent shift (`class_sep`)
      - Optional pseudo-class confounding via `n_pseudo` and `pseudo_effect`
      - Pure noise features

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    meta : DatasetMeta
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
    block_used: list[int] = []

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
                # Default: single block covering p_corr (fill remainder if user underspecified)
                block_used = [p_corr] if block_sizes is None else list(block_sizes) + [p_corr - sum(block_sizes)]
            else:
                block_used = list(block_sizes)
            corr = _make_block_corr(block_used, corr_within=corr_within, corr_between=corr_between)

        # Sample correlated base: mean 0, diag 1
        X_corr = rng.multivariate_normal(mean=np.zeros(p_corr), cov=corr, size=n_samples, check_valid="ignore")
        X_corr = X_corr.astype(np.float64, copy=False)

        # Indices within correlated part
        idx_inf = np.arange(0, n_informative, dtype=np.int64)
        idx_pseudo = np.arange(n_informative, n_informative + n_pseudo, dtype=np.int64)

        # Class separation for informative features
        if n_informative > 0 and class_sep != 0.0:
            # Shift magnitude: +/- class_sep/2 to produce total mean difference ≈ class_sep
            shift = float(class_sep) / 2.0
            s = np.where(y == 1, +shift, -shift).reshape(-1, 1)
            X_corr[:, idx_inf] = X_corr[:, idx_inf] + s

        # Pseudo-class shifts (independent confounding)
        if n_pseudo > 0 and pseudo_effect != 0.0 and z is not None:
            pshift = float(pseudo_effect) / 2.0
            s_p = np.where(z == 1, +pshift, -pshift).reshape(-1, 1)
            X_corr[:, idx_pseudo] = X_corr[:, idx_pseudo] + s_p

    # Pure noise features (independent of everything)
    X_noise = np.empty((n_samples, 0), dtype=np.float64)
    if p_noise > 0:
        if noise_dist not in {"normal", "uniform", "lognormal"}:
            raise ValueError("noise_dist must be 'normal', 'uniform', or 'lognormal'.")
        if noise_dist == "normal":
            X_noise = rng.normal(loc=0.0, scale=1.0, size=(n_samples, p_noise)).astype(np.float64)
        elif noise_dist == "uniform":
            # Uniform in [-1, 1] roughly matches variance scale of N(0,1)
            X_noise = rng.uniform(low=-1.0, high=1.0, size=(n_samples, p_noise)).astype(np.float64)
        else:  # lognormal
            # Heavier-tailed positive noise; sigma chosen moderate to avoid huge means
            X_noise = rng.lognormal(mean=0.0, sigma=0.5, size=(n_samples, p_noise)).astype(np.float64)

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
