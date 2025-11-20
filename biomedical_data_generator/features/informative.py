# informative.py
# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Generation of *free* informative features and class separation.

This module is responsible only for:

- building class labels y from DatasetConfig.class_configs,
- sampling base values for free informative features according to the
  per-class distributions (ClassConfig),
- applying class-wise mean shifts controlled by DatasetConfig.class_sep.

Correlated clusters (including anchors) are handled entirely in
`correlated.py`. Noise features are handled in `noise.py`.

The shifting logic is implemented in `shift_classes` and can also be
re-used by other modules (e.g. for anchor effects in correlated clusters).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from biomedical_data_generator.config import AnchorMode, DatasetConfig
from biomedical_data_generator.utils.sampling import sample_2d_array

__all__ = [
    "generate_informative_features",
    "shift_classes",
]


# ---------------------------------------------------------------------------
# Helpers for class separation
# ---------------------------------------------------------------------------


def _normalize_class_sep(
    class_sep: float | Sequence[float],
    K: int,
) -> np.ndarray:
    """Return a (K-1,)-vector of separations between neighbouring classes."""
    if K <= 1:
        raise ValueError(f"_normalize_class_sep: K must be >= 2, got {K}.")

    # Scalar → broadcast
    if np.isscalar(class_sep):
        v = float(class_sep)
        if not np.isfinite(v):
            raise ValueError(f"class_sep must be finite, got {class_sep!r}.")
        return np.full(K - 1, v, dtype=float)

    # Sequence → numeric vector of length (K-1)
    sep_vec = np.asarray(class_sep, dtype=float)
    if sep_vec.ndim != 1:
        raise ValueError(f"class_sep must be 1D, got shape {sep_vec.shape}.")
    if sep_vec.shape[0] != K - 1:
        raise ValueError(f"class_sep length must be K-1 (={K-1}), got {sep_vec.shape[0]}.")
    if not np.all(np.isfinite(sep_vec)):
        raise ValueError("class_sep entries must be finite numbers.")
    return sep_vec


def _class_offsets_from_sep(sep_vec: np.ndarray) -> np.ndarray:
    """Construct centered class-wise offsets from a (K-1,)-separation vector."""
    K = sep_vec.shape[0] + 1
    mu = np.empty(K, dtype=float)
    mu[0] = 0.0
    for k in range(1, K):
        mu[k] = mu[k - 1] + sep_vec[k - 1]
    mu -= mu.mean()
    return mu


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------


def _build_class_labels(cfg: DatasetConfig) -> np.ndarray:
    """Build numeric class labels 0..K-1 from DatasetConfig.class_configs."""
    labels: list[np.ndarray] = []
    for idx, cls_cfg in enumerate(cfg.class_configs):
        labels.append(np.full(cls_cfg.n_samples, idx, dtype=int))
    y = np.concatenate(labels, axis=0)
    if y.shape[0] != cfg.n_samples:
        raise RuntimeError(f"Inconsistent label construction: got {y.shape[0]} labels, " f"expected {cfg.n_samples}.")
    return y


# ---------------------------------------------------------------------------
# Core shifting logic (also reusable by correlated.py for anchors)
# ---------------------------------------------------------------------------


def shift_classes(
    X: np.ndarray,
    y: np.ndarray,
    *,
    informative_idx: Iterable[int],
    anchor_contrib: Mapping[int, tuple[float, int]] | None = None,
    class_sep: float | Sequence[float] = 1.0,
    anchor_strength: float = 1.0,
    anchor_mode: AnchorMode = "equalized",  # "equalized" or "strong"
    spread_non_anchors: bool = True,
) -> None:
    """Apply class-wise mean shifts to informative features and optional anchors.

    This function modifies X *in place*.

    Parameters
    ----------
    X:
        Array of shape (n_samples, n_features).
    y:
        Array of shape (n_samples,) with class labels in {0, ..., K-1}.
    informative_idx:
        Indices of informative (non-anchor) features.
        Any index present in `anchor_contrib` is skipped here and treated
        as an anchor instead.
    anchor_contrib:
        Optional mapping: col -> (beta, cls_target)

        - col: column index of the anchor feature.
        - beta: per-anchor effect multiplier (e.g. derived from
          CorrClusterConfig.resolve_anchor_effect_size()).
        - cls_target: index of the "disease" / target class for the
          one-vs-rest shift.

        For the free informative stage (`informative.py`), this argument
        is typically None. In the correlated stage, it can be used to
        implement one-vs-rest anchors.

    class_sep:
        Scalar or sequence controlling the multi-class separation and
        providing a scale for anchors.

        - If scalar: uniform separation between neighbouring classes.
        - If sequence: length (K-1), where entry j approximates
              mu_{j+1} - mu_j
          for the non-anchor informative features.

    anchor_strength:
        Global multiplicative factor for all anchors (used only if
        anchor_contrib is not None).

    anchor_mode:
        - "equalized": anchor effect size is roughly invariant in K.
        - "strong": anchor effect grows with K and can dominate non-anchors.

    spread_non_anchors:
        If True (default), non-anchor informative features receive the
        multi-class offsets defined by class_sep. If False, they are left
        unchanged and only anchors are shifted.
    """
    if X.size == 0:
        return

    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be a 1D label array, got shape {y.shape}.")

    if y.size == 0:
        return

    K = int(y.max()) + 1
    if K <= 1:
        return

    # 1) Build class-wise offsets from class_sep -----------------------------
    sep_vec = _normalize_class_sep(class_sep, K)
    class_offsets = _class_offsets_from_sep(sep_vec)

    # sep_scale: scalar strength representative of the pairwise separations
    sep_scale = float(np.mean(np.abs(sep_vec))) if K > 1 else float(sep_vec[0])

    anchor_cols = set(anchor_contrib.keys()) if anchor_contrib else set()

    # 2) Spread for non-anchor informative features --------------------------
    if spread_non_anchors:
        for idx in informative_idx:
            if idx in anchor_cols:
                continue  # handled in anchor loop
            for k in range(K):
                X[y == k, idx] += class_offsets[k]

    # 3) Anchors: one-vs-rest pattern with K-invariant / strong scaling ------
    if anchor_contrib:
        for col, (beta, cls_target) in anchor_contrib.items():
            if not (0 <= cls_target < K):
                raise ValueError(f"anchor_contrib: cls_target={cls_target} out of range for K={K}.")

            beta_val = float(beta)

            if anchor_mode == "equalized":
                # Δμ(target vs. others) ≈ sep_scale * anchor_strength * beta
                A = sep_scale * anchor_strength * beta_val * (K - 1) / K
            elif anchor_mode == "strong":
                # Stronger growth with K; Δμ grows ~ K
                A = sep_scale * anchor_strength * beta_val * (K - 1) / 2.0
            else:
                raise ValueError(f"Unknown anchor_mode={anchor_mode!r}.")

            # Increase target class, decrease all others so that the global
            # mean of the column is unchanged.
            X[y == cls_target, col] += A
            for k in range(K):
                if k != cls_target:
                    X[y == k, col] -= A / (K - 1)


# ---------------------------------------------------------------------------
# Public API: generate_informative_features
# ---------------------------------------------------------------------------


def generate_informative_features(
    cfg: DatasetConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate all *free* informative features (no anchors, no clusters).

    This function encapsulates the "informative" stage for independent
    informative features:

    1. Builds numeric labels y from cfg.class_configs.
    2. Allocates a matrix X_inf of shape (n_samples, n_informative_free).
    3. Samples base values for each class using ClassConfig.class_distribution
       and class_distribution_params via `sample_2d_array`.
    4. Applies class-wise mean shifts as defined in `shift_classes` for all
       informative columns (multi-class offsets only; no anchors).

    The generator and DatasetMeta can derive all structural information
    (block position, counts, indices) directly from `DatasetConfig`
    (e.g. `cfg.n_informative_free`, `cfg.n_samples`).

    Parameters
    ----------
    cfg:
        DatasetConfig with validated fields and derived quantities.
    rng:
        NumPy random Generator (global dataset RNG).

    Returns:
    -------
    X_inf:
        Array of shape (n_samples, n_informative_free) with all free
        informative features.
    y:
        Array of shape (n_samples,) with class labels in {0, ..., K-1}.
    """
    n_samples = cfg.n_samples
    n_inf = int(cfg.n_informative_free)  # excludes informative anchors by design

    # 1) Build numeric class labels
    y = _build_class_labels(cfg)

    # Corner case: no informative features → return empty matrix
    if n_inf == 0:
        X_empty = np.empty((n_samples, 0), dtype=float)
        return X_empty, y

    # 2) Sample base values per class (before mean shifts)
    X_inf = np.empty((n_samples, n_inf), dtype=float)

    start = 0
    for cls_cfg in cfg.class_configs:
        n_cls = cls_cfg.n_samples
        stop = start + n_cls

        block = sample_2d_array(
            distribution=cls_cfg.class_distribution,
            params=cls_cfg.class_distribution_params,
            rng=rng,
            size=(n_cls, n_inf),
        )
        X_inf[start:stop, :] = block
        start = stop

    # 3) Apply class-wise mean shifts to all informative columns
    informative_idx = np.arange(n_inf, dtype=int)
    shift_classes(
        X_inf,
        y,
        informative_idx=informative_idx,
        anchor_contrib=None,  # no anchors in this stage
        class_sep=cfg.class_sep,  # already normalised in DatasetConfig
        # anchor_strength / anchor_mode defaults are unused here
        spread_non_anchors=True,
    )

    return X_inf, y
