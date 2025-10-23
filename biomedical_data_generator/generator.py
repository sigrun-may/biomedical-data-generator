# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Generator for synthetic classification datasets with correlated feature clusters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame

from .config import DatasetConfig, NoiseDistribution
from .features.correlated import sample_cluster
from .features.informative import shift_classes
from .features.noise import sample_noise
from .meta import DatasetMeta


def _labels_from_counts(counts: dict[int, int], n_classes: int, rng: np.random.Generator) -> np.ndarray:
    """Build label vector y exactly matching the requested counts and shuffle it.

    Args:
        counts: Mapping class -> count (non-negative integers).
        n_classes: Number of classes (must be >= 2).
        rng: Random number generator for shuffling.

    Returns:
    -------
        np.ndarray: Label vector of shape (sum(counts.values()),) with values in {0, ..., n_classes-1}.

    Raises:
    ------
        ValueError: If any count is negative or if counts keys are out of range.
        RuntimeError: If the assembled label vector size does not match the sum of counts.
    """
    y_parts = []
    for k in range(n_classes):
        c = int(counts.get(k, 0))
        if c < 0:
            raise ValueError("class_counts must be non-negative.")
        if c > 0:
            y_parts.append(np.full(c, k, dtype=int))
    y = np.concatenate(y_parts) if y_parts else np.empty((0,), dtype=int)
    if y.size != sum(int(counts.get(k, 0)) for k in range(n_classes)):
        raise RuntimeError("Counts assembly mismatch.")
    rng.shuffle(y)
    return y


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
            cols = list(range(current, current + c.n_cluster_features))
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
                proxies_from_clusters += max(c.n_cluster_features - 1, 0)
            else:
                anchor_idx[cid] = None
                for k, col in enumerate(cols, start=1):
                    names.append(
                        f"{cfg.prefix_corr}{cid}_{k}" if cfg.feature_naming == "prefixed" else f"feature_{len(names)+1}"
                    )
                    pseudo_idx.append(col)
                proxies_from_clusters += c.n_cluster_features
            current += c.n_cluster_features

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

    Returns:
    -------
        dict: Resolved parameters for the specified distribution.

    Raises:
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
    """Generate an n-class classification dataset with optional correlated clusters.

    Features are ordered as: cluster features (anchors first within each cluster),
    then free informative features, then free pseudo features, then noise features.
    Labels y must be explicitly specified via `cfg.class_counts` (exact per-class sample counts).
    Class-wise shifts are then applied to informative features and cluster anchors
    (via `anchor_effect_size`) to create class separation.
    Reproducibility is controlled by `cfg.random_state` and optional per-cluster seeds.
    Feature names follow either a "prefixed" scheme (e.g., `i*`, `corr{cid}_k`, `p*`, `n*`)
    or a generic `feature_1..p`. The returned `meta` includes role masks, cluster indices,
    empirical class proportions, and the resolved configuration.

    Args:
        cfg (DatasetConfig): Configuration including feature counts, cluster layout, correlation
            parameters, naming policy, randomness controls, `n_classes`, and `class_counts`.
            The `class_counts` parameter is required and must be a dict mapping class indices
            to exact sample counts (e.g., {0: 50, 1: 50}).
        return_dataframe (bool, optional): If True (default), return `X` as a `pandas.DataFrame`
            with column names. If False, return `X` as a NumPy array in the same column order.
        **overrides: Optional config overrides merged into `cfg` (e.g., `n_samples=...`).

    Returns:
    -------
        tuple:
            - X (pandas.DataFrame | np.ndarray): Shape (n_samples, n_features). By default a DataFrame
              with feature names in canonical order (clusters → free informative → free pseudo → noise).
            - y (np.ndarray): Shape (n_samples,). Integer labels in {0, 1, ..., n_classes-1}.
            - meta (DatasetMeta): Metadata including role masks, cluster indices/labels, empirical class
              weights, and the resolved configuration.

    Raises:
    ------
        ValueError: If `class_counts` is not specified or if sum(class_counts) != n_samples.
    """
    if overrides:
        cfg = cfg.model_copy(update=overrides)

    if int(cfg.n_classes) < 2:
        raise ValueError("n_classes must be >= 2.")

    # Require explicit class_counts
    if cfg.class_counts is None:
        raise ValueError(
            "class_counts must be specified. " "Specify exact per-class sample counts as a dict (e.g., {0: 50, 1: 50})."
        )

    # Validate class_counts
    if sum(cfg.class_counts.values()) != cfg.n_samples:
        raise ValueError(
            f"sum(class_counts) must equal n_samples. "
            f"Got sum={sum(cfg.class_counts.values())}, expected n_samples={cfg.n_samples}."
        )

    rng_global = np.random.default_rng(cfg.random_state)

    # STEP 1: Generate labels FIRST (before features, so class_rho can use them)
    K = int(cfg.n_classes)
    y = _labels_from_counts(cfg.class_counts, cfg.n_classes, rng_global)  # shape (n_samples,), dtype=int

    # names & roles (+ totals validation inside)
    names, inf_idx, pse_idx, noi_idx, cluster_idx, anch_idx = _make_names_and_roles(cfg)

    # STEP 2: Build features (now with access to labels for class_rho)
    # 2a) build matrices per cluster (respect per-cluster seed if provided)
    cluster_matrices: list[NDArray[np.float64]] = []
    # Map: anchor feature column -> (beta, class_id)
    anchor_contrib: dict[int, tuple[float, int]] = {}
    anchor_target_cls_map: dict[int, int | None] = {}
    cluster_label_map: dict[int, str | None] = {}
    col_start = 0
    if cfg.corr_clusters:
        for cid, c in enumerate(cfg.corr_clusters, start=1):
            seed = c.random_state if c.random_state is not None else cfg.random_state
            rng = np.random.default_rng(seed)
            B = sample_cluster(
                n_samples=cfg.n_samples,
                n_features=c.n_cluster_features,
                rng=rng,
                structure=c.structure,
                rho=c.rho,
                class_labels=y,  # NOW labels are available for class_rho!
                class_rho=c.class_rho,
                baseline_rho=c.rho_baseline,
            )
            # weak global coupling (same g for all cluster columns)
            if cfg.corr_between > 0.0:
                g = rng_global.normal(0.0, 1.0, size=(cfg.n_samples, 1))
                B = B + np.sqrt(cfg.corr_between) * g
            if c.anchor_role == "informative":
                anchor_col = col_start  # first column of this cluster in global X
                anchor_cls = 0 if c.anchor_class is None else int(c.anchor_class)
                if not (0 <= anchor_cls < cfg.n_classes):
                    raise ValueError(f"anchor_class {anchor_cls} out of range for n_classes={cfg.n_classes}.")
                beta = c.resolve_anchor_effect_size()  # Convert "small"/"medium"/"large" -> numeric
                anchor_contrib[anchor_col] = (beta, anchor_cls)
                anchor_target_cls_map[cid] = anchor_cls
            else:
                anchor_target_cls_map[cid] = None
            cluster_label_map[cid] = c.label
            cluster_matrices.append(B)
            col_start += c.n_cluster_features

    X_clusters = np.concatenate(cluster_matrices, axis=1) if cluster_matrices else np.empty((cfg.n_samples, 0))

    # 2b) free informative (exactly n_informative - n_anchors)
    n_anchors = sum(1 for c in (cfg.corr_clusters or []) if c.anchor_role == "informative")
    n_inf_free = cfg.n_informative - n_anchors
    X_inf = rng_global.normal(size=(cfg.n_samples, n_inf_free)) if n_inf_free > 0 else np.empty((cfg.n_samples, 0))

    # 2c) free pseudo (exactly cfg.n_pseudo, independent of proxies)
    X_pseudo = (
        rng_global.normal(size=(cfg.n_samples, cfg.n_pseudo)) if cfg.n_pseudo > 0 else np.empty((cfg.n_samples, 0))
    )

    # 2d) noise
    if cfg.n_noise > 0:
        params = _resolve_noise_params(cfg.noise_distribution, cfg.noise_scale, cfg.noise_params)
        distribution = (
            cfg.noise_distribution.value if hasattr(cfg.noise_distribution, "value") else str(cfg.noise_distribution)
        )
        noise_cols = np.stack(
            [
                sample_noise(
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

    # STEP 3: Compute empirical class stats (labels already generated in STEP 1)
    counts = np.bincount(y, minlength=K).astype(int)
    y_counts = {int(k): int(counts[k]) for k in range(K)}
    y_weights = tuple((counts.astype(float) / float(counts.sum())).tolist())

    # per-cluster role/beta maps for meta
    anchor_role_map: dict[int, str] = {}
    anchor_effect_size_map: dict[int, float] = {}
    if cfg.corr_clusters:
        i = 0
        for cid, c in enumerate(cfg.corr_clusters, start=1):
            anchor_role_map[cid] = c.anchor_role
            anchor_effect_size_map[cid] = c.resolve_anchor_effect_size() if c.anchor_role == "informative" else 0.0
            i += c.n_cluster_features

    # Shift feature values for classes
    shift_classes(
        X,
        y,
        informative_idx=inf_idx,
        anchor_contrib=anchor_contrib,  # do not gate anchors with the spread flag
        class_sep=float(cfg.class_sep),
        anchor_strength=float(getattr(cfg, "anchor_strength", 1.0)),
        anchor_mode=cfg.anchor_mode,
        spread_non_anchors=bool(getattr(cfg, "spread_non_anchors", True)),
    )

    # Final metadata
    meta = DatasetMeta(
        feature_names=names,
        informative_idx=inf_idx,
        pseudo_idx=pse_idx,
        noise_idx=noi_idx,
        corr_cluster_indices=cluster_idx,
        anchor_idx=anch_idx,
        anchor_role=anchor_role_map,
        anchor_effect_size=anchor_effect_size_map,
        anchor_target_cls=anchor_target_cls_map,
        cluster_label=cluster_label_map,
        y_weights=y_weights,
        y_counts=y_counts,
        n_classes=K,
        class_sep=float(cfg.class_sep),
        anchor_strength=float(getattr(cfg, "anchor_strength", 1.0)),
        anchor_mode=cfg.anchor_mode,
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
# Dataset-level acceptance helpers
# ==========================================================
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

    Returns:
    -------
        tuple:
            - seed (int): The selected seed.
            - X (pandas.DataFrame | np.ndarray): Feature matrix for that seed.
            - y (np.ndarray): Labels for that seed.
            - meta (DatasetMeta): Metadata for that seed.
            - score (float): The achieved score.

    Raises:
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
