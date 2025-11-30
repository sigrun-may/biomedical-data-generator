# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Generator for synthetic classification datasets with correlated feature clusters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DatasetConfig
from .effects.batch import apply_batch_effects_from_config
from .features.correlated import sample_all_correlated_clusters
from .features.informative import generate_informative_features
from .meta import DatasetMeta
from .utils.sampling import sample_distribution


def _make_names_and_roles(
    cfg: DatasetConfig,
    *,
    n_cluster_cols: int,
    n_inf_cols: int,
    n_noise_cols: int,
) -> tuple[
    list[str],  # names
    list[int],  # informative_idx (anchors + free informative)
    list[int],  # noise_idx (independent/free noise only)
    dict[int, list[int]],  # cluster_indices[cid] -> list of column indices
    dict[int, int | None],  # anchor_idx[cid] -> anchor column (or None)
]:
    """Build feature names and role indices for the final concatenated matrix.

    We assume that :func:`generate_dataset` has constructed the feature
    matrix ``x`` by horizontal concatenation in the following order::

        x = [x_informative | x_clusters | x_noise]

    where

    * ``x_informative`` contains **only free informative features**
      (no cluster anchors),
    * ``x_clusters`` contains, for each :class:`CorrClusterConfig`,
      the **anchor** (first column of the block) followed by its
      **proxy features**,
    * ``x_noise`` contains **only independent/free noise features**
      (no cluster anchors).

    This function is purely book-keeping:

    * it assigns human-readable feature names,
    * it returns index lists for informative and (independent) noise
      features,
    * it provides cluster layouts via ``cluster_indices`` and
      ``anchor_idx`` so that proxy indices can be derived later as
      ``set(cluster_indices[cid]) - {anchor_idx[cid]}``.

    Args:
        cfg: Resolved :class:`DatasetConfig` used for generation.
        n_cluster_cols: Number of columns in ``x_clusters``.
        n_inf_cols: Number of columns in ``x_informative``.
        n_noise_cols: Number of columns in ``x_noise``.

    Returns:
        names, informative_idx, noise_idx, cluster_indices, anchor_idx
    """
    names: list[str] = []
    informative_idx: list[int] = []
    noise_idx: list[int] = []
    cluster_indices: dict[int, list[int]] = {}
    anchor_idx: dict[int, int | None] = {}

    # -------------------------------------------------------------
    # Sanity checks: shapes from generator vs. structural config
    # -------------------------------------------------------------
    clusters = cfg.corr_clusters or []
    expected_cluster_cols = sum(int(c.n_cluster_features) for c in clusters)
    if n_cluster_cols != expected_cluster_cols:
        raise ValueError(
            "Mismatch between x_clusters.shape[1] and corr_clusters definition: "
            f"x_clusters has {n_cluster_cols} columns, but corr_clusters imply "
            f"{expected_cluster_cols} columns."
        )

    # Expected number of free informative / noise features from config
    n_inf_free_expected = cfg.n_informative_free
    n_noise_free_expected = cfg.n_noise_free

    if n_inf_cols != n_inf_free_expected:
        raise ValueError(
            "generate_informative_features must produce exactly "
            f"cfg.n_informative_free={n_inf_free_expected} columns, "
            f"but returned {n_inf_cols}."
        )

    if n_noise_cols != n_noise_free_expected:
        raise ValueError(
            "The noise block must contain exactly cfg.n_noise_free "
            f"={n_noise_free_expected} independent noise features, "
            f"but x_noise has {n_noise_cols} columns."
        )

    total_cols = n_inf_cols + n_cluster_cols + n_noise_cols
    if total_cols != cfg.n_features:
        raise ValueError(
            "Total number of columns in X does not match cfg.n_features. "
            f"Got {total_cols} columns from generator but cfg.n_features="
            f"{cfg.n_features}."
        )

    # -------------------------------------------------------------
    # 1) Free informative features: block [0, n_inf_cols)
    # -------------------------------------------------------------
    for j in range(n_inf_cols):
        col = j
        if cfg.prefixed_feature_naming:
            names.append(f"{cfg.prefix_informative}{j + 1}")
        else:
            names.append(f"feature_{len(names) + 1}")
        # All columns in x_informative are informative
        informative_idx.append(col)

    # -------------------------------------------------------------
    # 2) Correlated clusters: block [n_inf_cols, n_inf_cols + n_cluster_cols)
    #    One contiguous block per CorrClusterConfig, in config order.
    # -------------------------------------------------------------
    current = n_inf_cols
    for cid, cluster_cfg in enumerate(clusters):
        k = int(cluster_cfg.n_cluster_features)
        cols = list(range(current, current + k))
        cluster_indices[cid] = cols

        # Anchor is always the first column of the block
        anchor_col = cols[0]
        anchor_idx[cid] = anchor_col

        # Name anchor (display: 1-based with cid+1)
        if cfg.prefixed_feature_naming:
            if cluster_cfg.anchor_role == "informative":
                anchor_name = f"{cfg.prefix_corr}{cid + 1}_anchor"  # corr1_anchor, corr2_anchor, ...
            else:
                anchor_name = f"{cfg.prefix_corr}{cid + 1}_1"  # corr1_1, corr2_1, ...
        else:
            anchor_name = f"feature_{len(names) + 1}"
        names.append(anchor_name)

        # Mark anchor as informative if requested
        if cluster_cfg.anchor_role == "informative":
            informative_idx.append(anchor_col)

        # Name proxy features (never added to informative_idx / noise_idx)
        for offset, col in enumerate(cols[1:], start=2):
            if cfg.prefixed_feature_naming:
                proxy_name = f"{cfg.prefix_corr}{cid + 1}_{offset}"  # corr1_2, corr1_3, ...
            else:
                proxy_name = f"feature_{len(names) + 1}"
            names.append(proxy_name)

        current += k

    # -------------------------------------------------------------
    # 3) Independent / free noise: block at the end
    # -------------------------------------------------------------
    noise_start = n_inf_cols + n_cluster_cols
    for j in range(n_noise_cols):
        col = noise_start + j
        if cfg.prefixed_feature_naming:
            names.append(f"{cfg.prefix_noise}{j + 1}")
        else:
            names.append(f"feature_{len(names) + 1}")
        noise_idx.append(col)

    # Final consistency check
    if len(names) != total_cols:
        raise AssertionError(
            "Internal inconsistency in _make_names_and_roles: constructed "
            f"{len(names)} names, but expected {total_cols}."
        )

    # Informative feature count in X (anchors + free informative) must match cfg.n_informative
    if len(informative_idx) != cfg.n_informative:
        raise AssertionError(
            "Mismatch between cfg.n_informative and resolved informative indices: "
            f"cfg.n_informative={cfg.n_informative}, but informative_idx has "
            f"{len(informative_idx)} entries."
        )

    return names, informative_idx, noise_idx, cluster_indices, anchor_idx


# =================
# Public generator
# =================
def generate_dataset(cfg, return_dataframe=True) -> tuple[pd.DataFrame | np.ndarray, np.ndarray, DatasetMeta]:
    """Generate dataset with clean module boundaries.

    Args:
        cfg: Resolved :class:`DatasetConfig` for dataset generation.
        return_dataframe: If ``True``, return features as a :class:`pandas.DataFrame`
            with named columns. If ``False``, return as a NumPy array.

    Returns:
        tuple: A 3-tuple containing:

            - **x** (:class:`pandas.DataFrame` or :class:`numpy.ndarray`):
              Feature matrix of shape ``(n_samples, n_features)``. Each row represents one sample (e.g., patient),
              each column represents one feature (e.g., biomarker, gene expression value). When returned
              as DataFrame, column names depend on ``cfg.feature_naming``: "prefixed" (default)
              uses type-based prefixes (``i`` for informative, ``corr`` for correlated
              clusters, ``n`` for noise), yielding names like ``i1, corr1_anchor, n1``.
              "sequential" uses sequential numbering ``feature_1, feature_2, ...``.
            - **y** (:class:`numpy.ndarray`):
              Class labels of shape ``(n_samples,)`` with integer values
              ``0, 1, ..., n_classes-1``.
            - **meta** (:class:`DatasetMeta`):
              Metadata object containing feature masks (informative, correlated, noise,
              batch-specific), correlation block specifications, batch assignments,
              and complete generation configuration.
    """
    rng_global = np.random.default_rng(cfg.random_state)

    # ================================================================
    # STEP 1: Generate informative features + labels (with shifts)
    # ================================================================
    x_informative, y = generate_informative_features(cfg, rng_global)
    # Returns SHIFTED features (class separation already applied)

    # ================================================================
    # STEP 2: Generate correlated clusters (with anchor shifts)
    # ================================================================
    x_clusters, cluster_meta = sample_all_correlated_clusters(cfg=cfg, y=y, rng=rng_global)
    # Returns clusters with anchor shifts already applied

    # ================================================================
    # STEP 3: Generate noise features
    # ================================================================
    x_noise = sample_distribution(
        distribution=cfg.noise_distribution,
        params=cfg.noise_distribution_params,
        rng=rng_global,
        size=(cfg.n_samples, cfg.n_noise),
    )

    # ================================================================
    # STEP 4: Concatenate all feature blocks
    # ================================================================
    x = np.concatenate([x_informative, x_clusters, x_noise], axis=1)

    # ================================================================
    # STEP 5: Apply batch effects (technical overlay)
    # ================================================================
    batch_labels = None
    batch_effects = None
    if cfg.batch is not None and cfg.batch.n_batches > 1:
        x, batch_labels, batch_effects = apply_batch_effects_from_config(
            x=x,
            y=y,
            batch_config=cfg.batch,
            rng=rng_global,
        )

    # ================================================================
    # STEP 6: Build names and role indices (knows final structure)
    # ================================================================
    names, inf_idx, noi_idx, cluster_idx, anch_idx = _make_names_and_roles(
        cfg,
        n_cluster_cols=x_clusters.shape[1],
        n_inf_cols=x_informative.shape[1],
        n_noise_cols=x_noise.shape[1],
    )

    # ================================================================
    # STEP 7: Build metadata
    # ================================================================
    counts = np.bincount(y, minlength=cfg.n_classes)
    meta = DatasetMeta(
        feature_names=names,
        informative_idx=inf_idx,
        noise_idx=noi_idx,
        corr_cluster_indices=cluster_idx,
        anchor_idx=anch_idx,
        anchor_role=cluster_meta["anchor_role"],
        anchor_effect_size=cluster_meta["anchor_effect_size"],
        anchor_class=cluster_meta["anchor_class"],
        cluster_label=cluster_meta["label"],
        n_classes=cfg.n_classes,
        class_names=cfg.class_labels,
        samples_per_class={int(k): int(counts[k]) for k in range(cfg.n_classes)},
        class_sep=cfg.class_sep,
        corr_between=cfg.corr_between,
        batch_labels=batch_labels,
        batch_effects=batch_effects,
        batch_config=cfg.batch.model_dump() if cfg.batch is not None else None,
        random_state=cfg.random_state,
        resolved_config=cfg.model_dump(),
    )
    if return_dataframe:
        return pd.DataFrame(x, columns=names), y, meta
    return x, y, meta
