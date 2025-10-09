# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Generator for synthetic classification datasets with correlated feature clusters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame

from .config import CorrCluster, DatasetConfig, NoiseDistribution


# =========================
# Ground-truth meta
# =========================
@dataclass(frozen=True)
class DatasetMeta:
    """Metadata about the generated dataset."""

    feature_names: list[str]

    informative_idx: list[int]  # includes cluster anchors + free i*
    pseudo_idx: list[int]  # corr* proxies + free p*
    noise_idx: list[int]

    # Correlated cluster structure
    corr_cluster_indices: dict[int, list[int]]  # cluster_id -> column indices
    anchor_idx: dict[int, int | None]  # cluster_id -> anchor col (or None)
    anchor_role: dict[int, str]  # "informative" | "pseudo" | "noise"
    anchor_beta: dict[int, float]  # 0.0 if latent
    anchor_target_cls: dict[int, int | None] # target class for the anchor (one-vs-rest)
    cluster_label: dict[int, str | None]  # didactic tags per cluster

    # Class distribution
    y_weights: tuple[float, ...]
    y_counts: dict[int, int]

    # Provenance / signal settings
    n_classes: int
    class_sep: float
    corr_between: float

    # --- optional (with defaults) ---
    anchor_strength: float = 1.0
    anchor_mode: Literal["equalized", "strong"] = "equalized"
    spread_non_anchors: bool = True

    random_state: int | None = None
    resolved_config: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Convert to a dictionary (e.g., for JSON serialization)."""
        return asdict(self)


def _shift_classes(
    X: np.ndarray,
    y: np.ndarray,
    *,
    informative_idx: Iterable[int],
    anchor_contrib: Dict[int, Tuple[float, int]] | None = None,  # col -> (beta, cls)
    class_sep: float = 1.0,
    anchor_strength: float = 1.0,
    anchor_mode: str = "equalized",      # "equalized" or "strong"
    spread_non_anchors: bool = True,
) -> None:
    """Minimal, readable class-wise shifting with two knobs.

    Args:
        X: Feature matrix (modified in place).
        y: Class labels in {0, ..., K-1}.
        informative_idx: indices of informative (non-anchor) features.
        anchor_contrib: optional mapping col -> (beta, cls)
            - beta scales the anchor contribution per feature.
            - cls is the target class for the one-vs-rest anchor.
        class_sep: global separation scale for non-anchors (spread across classes).
        anchor_strength: additional scale for anchors (one-vs-rest).
        anchor_mode: "equalized" (default) or "strong".
            - "equalized": anchor strength is independent of K (recommended).
            - "strong": anchor strength grows with K (can overwhelm non-anchors).
        spread_non_anchors: if True (default), spread non-anchor informative features across classes.
    """
    if X.size == 0:
        return
    K = int(np.max(y)) + 1
    if K <= 1:
        return

    # 1) Spread for non-anchors
    if spread_non_anchors and K > 1:
        denom = K - 1
        spread_vec = class_sep * (np.arange(K, dtype=float) - (K - 1) / 2) / denom
        anchor_cols = set(anchor_contrib.keys()) if anchor_contrib else set()
        for idx in informative_idx:
            if idx in anchor_cols:
                continue
            for k in range(K):
                X[y == k, idx] += spread_vec[k]

    # 2) Anchors: one-vs-rest
    if anchor_contrib:
        for col, (beta, cls) in anchor_contrib.items():
            if anchor_mode == "equalized":
                A = class_sep * anchor_strength * float(beta) * (K - 1) / K
            elif anchor_mode == "strong":
                A = class_sep * anchor_strength * float(beta) * (K - 1) / 2
            else:
                raise ValueError(f"Unknown anchor_mode={anchor_mode!r}")
            X[y == cls, col] += A
            for k in range(K):
                if k != cls:
                    X[y == k, col] -= A / (K - 1)
    # # Determine number of classes
    # K = int(np.max(y)) + 1 if X.shape[0] > 0 else 0
    # if K <= 1 or class_sep == 0.0:
    #     return
    #
    # # 1) Spread for non-anchors (optional)
    # if spread_non_anchors and K > 1:
    #     denom = K - 1
    #     spread_vec = class_sep * (np.arange(K, dtype=float) - (K - 1) / 2) / denom
    #     anchor_cols = set(anchor_contrib.keys()) if anchor_contrib else set()
    #
    #     for idx in informative_idx:
    #         if idx in anchor_cols:
    #             continue
    #         for k in range(K):
    #             X[y == k, idx] += spread_vec[k]
    #
    # # 2) Anchors (one-vs-rest), K-invariant strength:
    # #    A = class_sep * anchor_strength * beta * (K-1) / K
    # if anchor_contrib:
    #     for col, (beta, cls) in anchor_contrib.items():
    #         A = class_sep * anchor_strength * float(beta) * (K - 1) / K
    #         X[y == cls, col] += A
    #         for k in range(K):
    #             if k != cls:
    #                 X[y == k, col] -= A / (K - 1)



def _sample_noise(
    rng: np.random.Generator, n: int, dist: str, scale: float, params: Mapping[str, Any] | None = None
) -> np.ndarray:
    """Sample n noise values from the specified distribution.

    Args:
        rng: Random number generator.
        n: Number of samples.
        dist: Distribution name ("normal", "uniform", "laplace").
        scale: Scale parameter (stddev for normal/laplace, half-width for uniform).
        params: Additional distribution-specific parameters.

    Returns
    -------
        np.ndarray: Array of shape (n,) with sampled noise values.

    Raises
    ------
        ValueError: If dist is unsupported or parameters are invalid.
    """
    params = dict(params or {})
    if dist == "normal":
        loc = float(params.pop("loc", 0.0))
        return rng.normal(loc=loc, scale=scale, size=n)
    elif dist == "uniform":
        low = float(params.pop("low", -scale))
        high = float(params.pop("high", scale))
        return rng.uniform(low=low, high=high, size=n)
    elif dist == "laplace":
        loc = float(params.pop("loc", 0.0))
        return rng.laplace(loc=loc, scale=scale, size=n)
    else:
        raise ValueError(f"Unsupported noise_distribution: {dist}")


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


def _sample_cluster_matrix(n: int, cluster: CorrCluster, rng: np.random.Generator) -> NDArray[np.float64]:
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
    random_state: int | None = None,
    label: str | None = None,
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Generate a single correlated feature cluster (no labels y involved).

    Returns (X_cluster, meta) where meta contains the empirical correlation matrix.

    Args:
        n_samples: Number of samples (rows).
        size: Number of features (columns).
        rho: Target correlation between features (0 ≤ rho < 1).
        structure: "equicorrelated" or "toeplitz".
        random_state: Random seed for reproducibility.
        label: Optional didactic tag for this cluster.

    Returns
    -------
        tuple:
            - X (np.ndarray): Shape (n_samples, size) with standardized columns.
            - meta (dict): Metadata with keys:
              size, rho, structure, random_state, label, corr_matrix (size x size),
              mean_offdiag, min_offdiag.

    Raises
    ------
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

    rng = np.random.default_rng(random_state)
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
        "random_state": random_state,
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

    Returns
    -------
        tuple:
            - seed (int): The first seed that satisfied the condition.
            - meta (dict): Metadata as returned by generate_correlated_cluster.

    Raises
    ------
        RuntimeError: If no seed satisfied the rule within max_tries.
        ValueError: If size < 1 or rho_target not in [0, 1).
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    if not (0.0 <= rho_target < 1.0):
        raise ValueError("rho_target must be in [0, 1)")

    seed = start_seed
    for _ in range(max_tries):
        _, m = generate_correlated_cluster(n_samples, size, rho_target, structure, random_state=seed)
        mean_off = cast(float, m["mean_offdiag"])
        min_off = cast(float, m["min_offdiag"])

        ok = False
        if tol is not None:
            ok = abs(mean_off - rho_target) <= tol
        if not ok:
            val = mean_off if metric == "mean_offdiag" else min_off
            ok = (val >= threshold) if op == ">=" else (val <= threshold)

        if ok:
            return seed, m
        seed += 1
    raise RuntimeError("No seed satisfied the correlation rule within max_tries.")


# ==================
# Naming & role map
# ==================
def _make_names_and_roles(
    cfg: DatasetConfig,
) -> tuple[list[str], list[int], list[int], list[int], dict[int, list[int]], dict[int, int | None]]:
    """Build feature names and role indices.

    Important semantics:
      - `n_pseudo` counts ONLY free pseudo-features (`p*`), NOT cluster proxies.
      - Cluster proxies (`corr{cid}_k`) are additional pseudo-features coming from clusters.
      - Therefore the expected total number of features is:
            n_features_expected = n_informative + n_pseudo + n_noise + proxies_from_clusters
    """
    names: list[str] = []
    informative_idx: list[int] = []
    pseudo_idx: list[int] = []
    noise_idx: list[int] = []
    cluster_indices: dict[int, list[int]] = {}
    anchor_idx: dict[int, int | None] = {}

    current = 0
    proxies_from_clusters = 0

    # 1) clusters first for contiguous columns
    if cfg.corr_clusters:
        for cid, c in enumerate(cfg.corr_clusters, start=1):
            cols = list(range(current, current + c.size))
            cluster_indices[cid] = cols
            if c.anchor_role == "informative":
                # first col is anchor -> named as informative
                anchor_col = cols[0]
                anchor_idx[cid] = anchor_col
                names.append(
                    f"{cfg.prefix_informative}{len(informative_idx)+1}"
                    if cfg.feature_naming == "prefixed"
                    else f"feature_{len(names)+1}"
                )
                informative_idx.append(anchor_col)
                # proxies (remaining columns in this cluster)
                for k, col in enumerate(cols[1:], start=2):
                    names.append(
                        f"{cfg.prefix_corr}{cid}_{k}" if cfg.feature_naming == "prefixed" else f"feature_{len(names)+1}"
                    )
                    pseudo_idx.append(col)
                proxies_from_clusters += max(c.size - 1, 0)
            else:
                anchor_idx[cid] = None
                for k, col in enumerate(cols, start=1):
                    names.append(
                        f"{cfg.prefix_corr}{cid}_{k}" if cfg.feature_naming == "prefixed" else f"feature_{len(names)+1}"
                    )
                    pseudo_idx.append(col)
                proxies_from_clusters += c.size
            current += c.size

    # 2) free informative outside clusters
    n_anchors = sum(1 for c in (cfg.corr_clusters or []) if c.anchor_role == "informative")
    if cfg.n_informative < n_anchors:
        raise ValueError(f"n_informative ({cfg.n_informative}) < number of informative anchors ({n_anchors}).")
    n_inf_free = cfg.n_informative - n_anchors
    for _ in range(n_inf_free):
        names.append(
            f"{cfg.prefix_informative}{len(informative_idx)+1}"
            if cfg.feature_naming == "prefixed"
            else f"feature_{len(names)+1}"
        )
        informative_idx.append(len(names) - 1)

    # 3) free pseudo (exactly cfg.n_pseudo, independent of proxies)
    for j in range(cfg.n_pseudo):
        names.append(f"{cfg.prefix_pseudo}{j+1}" if cfg.feature_naming == "prefixed" else f"feature_{len(names)+1}")
        pseudo_idx.append(len(names) - 1)

    # 4) noise
    for j in range(cfg.n_noise):
        names.append(f"{cfg.prefix_noise}{j+1}" if cfg.feature_naming == "prefixed" else f"feature_{len(names)+1}")
        noise_idx.append(len(names) - 1)

    # Totals validation with proxies added on top
    n_features_expected = cfg.n_informative + cfg.n_pseudo + cfg.n_noise + proxies_from_clusters
    if len(names) != n_features_expected:
        raise AssertionError((len(names), n_features_expected))
    if cfg.n_features != n_features_expected:
        raise ValueError(
            "cfg.n_features must equal n_informative + n_pseudo + n_noise + proxies_from_clusters "
            f"= {cfg.n_informative} + {cfg.n_pseudo} + {cfg.n_noise} + {proxies_from_clusters} "
            f"= {n_features_expected}, but got n_features={cfg.n_features}."
        )

    return names, informative_idx, pseudo_idx, noise_idx, cluster_indices, anchor_idx


# ================
# Softmax utility
# ================
def _softmax(Z: NDArray[np.float64]) -> NDArray[np.float64]:
    Z = Z - Z.max(axis=1, keepdims=True)
    np.exp(Z, out=Z)
    Z_sum = Z.sum(axis=1, keepdims=True)
    Z /= Z_sum
    return Z


def _resolve_noise_params(dist: str, noise_scale: float, noise_params: Mapping[str, Any] | None) -> dict[str, float]:
    """Resolve noise distribution parameters with defaults.

    Returns a params dict that _always_ includes the required keys for the chosen dist.
    - normal/laplace: {'loc', 'scale'}
    - uniform: {'low', 'high'}
    Any keys given in noise_params override these defaults.

    Args:
        dist: Distribution name ("normal", "uniform", "laplace").
        noise_scale: Scale parameter (stddev for normal/laplace, half-width for uniform).
        noise_params: Additional distribution-specific parameters.

    Returns
    -------
        dict: Resolved parameters for the specified distribution.

    Raises
    ------
        ValueError: If dist is unsupported.
    """
    # normalize
    if isinstance(dist, NoiseDistribution):
        key = dist.value
    else:
        key = str(dist)
        # handle accidental Enum stringification like "NoiseDistribution.uniform"
        if key.startswith("NoiseDistribution."):
            key = key.split(".", 1)[1]
        key = key.lower()

    if key == "normal":
        params = {"loc": 0.0, "scale": float(noise_scale)}
        if noise_params:
            params.update({k: float(v) for k, v in noise_params.items() if k in ("loc", "scale")})
        return params

    if key == "laplace":
        params = {"loc": 0.0, "scale": float(noise_scale)}
        if noise_params:
            params.update({k: float(v) for k, v in noise_params.items() if k in ("loc", "scale")})
        return params

    if key == "uniform":
        # default to symmetric interval around 0 with width 2*scale unless params given
        if noise_params and {"low", "high"} <= set(noise_params.keys()):
            low = float(noise_params["low"])
            high = float(noise_params["high"])
            if not (low < high):
                raise ValueError("For uniform noise, require low < high.")
            return {"low": low, "high": high}
        s = float(noise_scale)
        return {"low": -s, "high": s}

    raise ValueError(f"Unsupported noise_distribution: {dist}")


# =================
# Public generator
# =================
def generate_dataset(
    cfg: DatasetConfig,
    /,
    *,
    return_dataframe: bool = True,
    **overrides,
) -> tuple[DataFrame | NDArray[np.float64], NDArray[np.int64], DatasetMeta]:
    """Generate an n-class (softmax) classification dataset with optional correlated clusters.

    Features are ordered as: cluster features (anchors first within each cluster),
    then free informative features, then free pseudo features, then noise features.
    Labels y are sampled from a softmax model over `n_classes`: cluster anchors add a positive
    contribution to their assigned class (via `anchor_beta`), free informative features contribute
    to classes in a simple round-robin assignment (β=0.8). Pseudo and noise features have β=0.
    Reproducibility is controlled by `cfg.random_state` and optional per-cluster seeds. If `weights`
    are specified (length `n_classes`), per-class intercepts are added so that
    softmax(mean_logits + intercepts) ≈ weights. Feature names follow either a “prefixed” scheme
    (e.g., `i*`, `corr{cid}_k`, `p*`, `n*`) or a generic `feature_1..p`. The returned `meta` includes
    role masks, cluster indices, empirical class proportions, and the resolved configuration.

    Args:
        cfg (DatasetConfig): Configuration including feature counts, cluster layout, correlation
            parameters, naming policy, randomness controls, `n_classes`, and optional `weights`.
        return_dataframe (bool, optional): If True (default), return `X` as a `pandas.DataFrame`
            with column names. If False, return `X` as a NumPy array in the same column order.
        **overrides: Optional config overrides merged into `cfg` (e.g., `n_samples=...`).

    Returns
    -------
        tuple:
            - X (pandas.DataFrame | np.ndarray): Shape (n_samples, n_features). By default a DataFrame
              with feature names in canonical order (clusters → free informative → free pseudo → noise).
            - y (np.ndarray): Shape (n_samples,). Integer labels in {0, 1, ..., n_classes-1}.
            - meta (DatasetMeta): Metadata including role masks, cluster indices/labels, empirical class
              weights, and the resolved configuration.
    """
    if overrides:
        cfg = cfg.model_copy(update=overrides)

    if int(cfg.n_classes) < 2:
        raise ValueError("n_classes must be >= 2.")
    if cfg.weights is not None:
        if len(cfg.weights) != cfg.n_classes:
            raise ValueError("weights must have length n_classes.")
        if any(w < 0 for w in cfg.weights):
            raise ValueError("weights must be non-negative.")
        wsum = float(sum(cfg.weights))
        if wsum <= 0:
            raise ValueError("sum(weights) must be > 0.")

    rng_global = np.random.default_rng(cfg.random_state)
    anchor_contrib: dict[int, tuple[float, int]] = {}  # col -> (beta, cls)
    cluster_label_map: dict[int, str | None] = {}
    anchor_target_cls_map: dict[int, int | None] = {}

    # names & roles (+ totals validation inside)
    names, inf_idx, pse_idx, noi_idx, cluster_idx, anch_idx = _make_names_and_roles(cfg)

    # 1) build matrices per cluster (respect per-cluster seed if provided)
    cluster_matrices: list[NDArray[np.float64]] = []
    # Map: anchor feature column -> (beta, class_id)
    anchor_contrib: dict[int, tuple[float, int]] = {}
    cluster_label_map: dict[int, str | None] = {}
    col_start = 0
    if cfg.corr_clusters:
        for cid, c in enumerate(cfg.corr_clusters, start=1):
            seed = c.random_state if c.random_state is not None else cfg.random_state
            rng = np.random.default_rng(seed)
            B = _sample_cluster_matrix(cfg.n_samples, c, rng)
            # weak global coupling (same g for all cluster columns)
            if cfg.corr_between > 0.0:
                g = rng_global.normal(0.0, 1.0, size=(cfg.n_samples, 1))
                B = B + np.sqrt(cfg.corr_between) * g
            if c.anchor_role == "informative":
                anchor_col = col_start  # first column of this cluster in global X
                anchor_cls = 0 if c.anchor_class is None else int(c.anchor_class)
                if not (0 <= anchor_cls < cfg.n_classes):
                    raise ValueError(
                        f"anchor_class {anchor_cls} out of range for n_classes={cfg.n_classes}."
                    )
                anchor_contrib[anchor_col] = (float(c.anchor_beta), anchor_cls)
                anchor_target_cls_map[cid] = anchor_cls
            else:
                anchor_target_cls_map[cid] = None
            cluster_label_map[cid] = c.label
            cluster_matrices.append(B)
            col_start += c.size
    X_clusters = np.concatenate(cluster_matrices, axis=1) if cluster_matrices else np.empty((cfg.n_samples, 0))

    # 2) free informative (exactly n_informative - n_anchors)
    n_anchors = sum(1 for c in (cfg.corr_clusters or []) if c.anchor_role == "informative")
    n_inf_free = cfg.n_informative - n_anchors
    X_inf = rng_global.normal(size=(cfg.n_samples, n_inf_free)) if n_inf_free > 0 else np.empty((cfg.n_samples, 0))

    # 3) free pseudo (exactly cfg.n_pseudo, independent of proxies)
    X_pseudo = (
        rng_global.normal(size=(cfg.n_samples, cfg.n_pseudo)) if cfg.n_pseudo > 0 else np.empty((cfg.n_samples, 0))
    )

    # 4) noise
    if cfg.n_noise > 0:
        params = _resolve_noise_params(cfg.noise_distribution, cfg.noise_scale, cfg.noise_params)
        distribution = (
            cfg.noise_distribution.value if hasattr(cfg.noise_distribution, "value") else str(cfg.noise_distribution)
        )
        noise_cols = np.stack(
            [
                _sample_noise(
                    rng_global,
                    n=cfg.n_samples,
                    dist=distribution,
                    scale=float(params.get("scale", cfg.noise_scale)),
                    params=params,
                )
                for _ in range(cfg.n_noise)
            ],
            axis=1,
        )
        X_noise = noise_cols
    else:
        X_noise = np.empty((cfg.n_samples, 0))

    # Concatenate in naming order: [clusters] + [free informative] + [free pseudo] + [noise]
    X = np.concatenate([X_clusters, X_inf, X_pseudo, X_noise], axis=1)

    # Check totals
    assert X.shape[1] == len(names) == cfg.n_features, (X.shape[1], len(names), cfg.n_features)

    # ----- logits for n_classes
    K = int(cfg.n_classes)
    logits = np.zeros((cfg.n_samples, K), dtype=float)

    # Anchors: add to their assigned class only
    for col, (beta, cls) in anchor_contrib.items():
        logits[:, cls] += beta * X[:, col]

    # Free informative: round-robin across classes with beta=0.8
    rr = 0
    for idx in inf_idx:
        # Skip anchor indices already handled above
        if idx in anchor_contrib:
            continue
        logits[:, rr] += 0.8 * X[:, idx]
        rr = (rr + 1) % K

    # Scale by class_sep
    logits *= float(cfg.class_sep)

    # Optional class prior matching via per-class intercepts
    if cfg.weights is not None:
        target = np.array(cfg.weights, dtype=float)
        target = target / target.sum()
        mean_logits = logits.mean(axis=0)  # shape (K,)
        with np.errstate(divide="ignore"):
            b = np.log(target) - mean_logits  # shift so softmax(mean+b) ≈ target
        logits = logits + b[None, :]

    # Sample labels
    P = _softmax(logits.copy())  # copy because _softmax is in-place
    cdf = P.cumsum(axis=1)
    r = rng_global.random(size=cfg.n_samples)[:, None]
    y = (r > cdf[:, :-1]).sum(axis=1).astype(np.int64)  # categorical draw

    # Empirical class stats
    y_counts = {int(k): int((y == k).sum()) for k in range(K)}
    y_weights = (
        tuple(float(y_counts[k] / cfg.n_samples) for k in range(K))
        if cfg.weights is None
        else tuple(float(w) / float(sum(cfg.weights)) for w in cfg.weights)
    )

    # per-cluster role/beta maps for meta
    anchor_role_map: dict[int, str] = {}
    anchor_beta_map: dict[int, float] = {}
    if cfg.corr_clusters:
        i = 0
        for cid, c in enumerate(cfg.corr_clusters, start=1):
            anchor_role_map[cid] = c.anchor_role
            anchor_beta_map[cid] = c.anchor_beta if c.anchor_role == "informative" else 0.0
            i += c.size

    # Shift feature values for classes
    _shift_classes(
        X,
        y,
        informative_idx=inf_idx,
        anchor_contrib=anchor_contrib,  # do not gate anchors with the spread flag
        class_sep=float(cfg.class_sep),
        anchor_strength=float(getattr(cfg, "anchor_strength", 1.0)),
        anchor_mode=str(getattr(cfg, "anchor_mode", "equalized")),
        spread_non_anchors=bool(getattr(cfg, "spread_non_anchors", True)),
    )
    #
    #
    # K = int(cfg.n_classes)
    # sep = float(cfg.class_sep)
    # for idx in inf_idx:
    #     if idx in anchor_contrib:
    #         continue
    #     # Spread means across classes
    #     for k in range(K):
    #         # Example: class 0 gets -sep/2, class 1 gets +sep/2
    #         shift = sep * (k - (K - 1) / 2) / (K - 1) if K > 1 else 0.0
    #         X[y == k, idx] += shift
    #
    # # Shift anchors strongly for better class separation
    # for col, (beta, cls) in anchor_contrib.items():
    #     shift = sep * beta * (K - 1) / 2  # strong shift towards assigned class
    #     for k in range(K):
    #         if k == cls:
    #             X[y == k, col] += shift
    #         else:
    #             X[y == k, col] -= shift / (K - 1)

    # Final metadata
    meta = DatasetMeta(
        feature_names=names,
        informative_idx=inf_idx,
        pseudo_idx=pse_idx,
        noise_idx=noi_idx,
        corr_cluster_indices=cluster_idx,
        anchor_idx=anch_idx,
        anchor_role=anchor_role_map,
        anchor_beta=anchor_beta_map,
        anchor_target_cls=anchor_target_cls_map,
        cluster_label=cluster_label_map,
        y_weights=y_weights,
        y_counts=y_counts,
        n_classes=K,
        class_sep=float(cfg.class_sep),
        anchor_strength=float(getattr(cfg, "anchor_strength", 1.0)),
        anchor_mode=str(getattr(cfg, "anchor_mode", "equalized")),
        spread_non_anchors=bool(getattr(cfg, "spread_non_anchors", True)),
        corr_between=float(cfg.corr_between),
        random_state=cfg.random_state,
        resolved_config=cfg.model_dump(),
    )

    if return_dataframe:
        X_df: DataFrame = pd.DataFrame(X, columns=names)
        return X_df, y, meta
    return X, y, meta


# ==========================================================
# Dataset-level acceptance helpers (multiclass-aware)
# ==========================================================
def find_dataset_seed_for_class_weights(
    cfg: DatasetConfig,
    /,
    *,
    tol: float = 0.02,
    start_seed: int = 0,
    max_tries: int = 500,
    return_dataframe: bool = True,
    **overrides,
) -> tuple[int, DataFrame | NDArray[np.float64], NDArray[np.int64], DatasetMeta]:
    """Find a random_state such that empirical class proportions approximate cfg.weights.

    Tries seeds starting at `start_seed` and returns the first seed for which the
    L1 distance between the empirical class proportions and the target weights is <= tol.
    Works for multiclass (n_classes >= 2). If cfg.weights is None, the target is uniform.

    Args:
        cfg: Base configuration to use (n_classes may be > 2).
        tol: L1 tolerance against target weights. Defaults to 0.02.
        start_seed: First seed to try. Defaults to 0.
        max_tries: Maximum number of seeds to try. Defaults to 500.
        return_dataframe: Return X as DataFrame (default) or ndarray.
        **overrides: Optional keyword overrides merged into cfg for generation.

    Returns
    -------
        tuple:
            - seed (int): The seed that satisfied the tolerance.
            - X (pandas.DataFrame | np.ndarray): Feature matrix for that seed.
            - y (np.ndarray): Labels for that seed.
            - meta (DatasetMeta): Metadata for that seed.

    Raises
    ------
        RuntimeError: If no seed satisfies the tolerance within max_tries.
        ValueError: If cfg.n_classes < 2 or weights length mismatches n_classes.
    """
    if cfg.n_classes < 2:
        raise ValueError("n_classes must be >= 2.")

    # Normalize/derive target weights
    if cfg.weights is None:
        target = np.full(cfg.n_classes, 1.0 / cfg.n_classes, dtype=float)
    else:
        if len(cfg.weights) != cfg.n_classes:
            raise ValueError("weights must have length n_classes.")
        target = np.asarray(cfg.weights, dtype=float)
        s = target.sum()
        if s <= 0:
            raise ValueError("sum(weights) must be > 0.")
        target = target / s

    seed = start_seed
    for _ in range(max_tries):
        X, y, meta = generate_dataset(cfg, return_dataframe=return_dataframe, **({"random_state": seed} | overrides))
        # empirical proportions
        counts = np.bincount(y, minlength=cfg.n_classes).astype(float)
        emp = counts / counts.sum()
        if np.abs(emp - target).sum() <= tol:
            return seed, X, y, meta
        seed += 1
    raise RuntimeError("No dataset seed satisfied class-weights tolerance within max_tries.")


def find_dataset_seed_for_score(
    cfg: DatasetConfig,
    scorer: Callable[[DataFrame | NDArray[np.float64], NDArray[np.int64], DatasetMeta], float],
    /,
    *,
    mode: Literal["max", "min"] = "max",
    threshold: float | None = None,
    start_seed: int = 0,
    max_tries: int = 200,
    return_dataframe: bool = True,
    **overrides,
) -> tuple[int, DataFrame | NDArray[np.float64], NDArray[np.int64], DatasetMeta, float]:
    """Find a random_state that optimizes a user-provided scorer(X, y, meta).

    Tries seeds starting at `start_seed`. If `threshold` is given, stops as soon as it
    finds a seed where the score meets the threshold (>= for mode="max", <= for "min").
    Otherwise, returns the best-scoring seed after max_tries.

    Args:
        cfg: Base configuration.
        scorer: Function scorer(X, y, meta) -> float.
        mode: Optimize to "max" or "min". Defaults to "max".
        threshold: Early-stop threshold (>= if mode="max", <= if "min").
        start_seed: First seed to try. Defaults to 0.
        max_tries: Maximum number of seeds. Defaults to 200.
        return_dataframe: Return X as DataFrame (default) or ndarray.
        **overrides: Optional keyword overrides merged into cfg for generation.

    Returns
    -------
        tuple:
            - seed (int): The selected seed.
            - X (pandas.DataFrame | np.ndarray): Feature matrix for that seed.
            - y (np.ndarray): Labels for that seed.
            - meta (DatasetMeta): Metadata for that seed.
            - score (float): The achieved score.

    Raises
    ------
        ValueError: If mode is invalid.
        RuntimeError: If threshold is given but not met within max_tries.
    """
    if mode not in ("max", "min"):
        raise ValueError('mode must be "max" or "min".')

    best: tuple[float, int, DataFrame | NDArray[np.float64], NDArray[np.int64], DatasetMeta] | None = None
    cmp = (lambda a, b: a > b) if mode == "max" else (lambda a, b: a < b)

    seed = start_seed
    for _ in range(max_tries):
        X, y, meta = generate_dataset(cfg, return_dataframe=return_dataframe, **({"random_state": seed} | overrides))
        s = float(scorer(X, y, meta))

        if best is None or cmp(s, best[0]):
            best = (s, seed, X, y, meta)

        if threshold is not None:
            if (mode == "max" and s >= threshold) or (mode == "min" and s <= threshold):
                return seed, X, y, meta, s
        seed += 1

    if threshold is not None:
        raise RuntimeError("No dataset seed met the threshold within max_tries.")
    assert best is not None
    s, seed, X, y, meta = best

    return seed, X, y, meta, s
