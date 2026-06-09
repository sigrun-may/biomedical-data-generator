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
from .features.informative import generate_informative_features, resolve_standalone_groups
from .meta import BatchMeta, DatasetMeta, StandaloneGroupMeta
from .utils.sampling import sample_distribution


def _make_names_and_roles(
    cfg: DatasetConfig,
    *,
    n_cluster_cols: int,
    n_inf_cols: int,
    n_noise_cols: int,
) -> tuple[
    list[str],  # names
    list[int],  # informative_idx (exhaustive informative partition)
    list[int],  # noise_idx (exhaustive noise partition)
    dict[int, list[int]],  # cluster_indices[cid] -> list of column indices
    dict[int, int],  # anchor_idx[cid] -> structural anchor column (block start + anchor_index)
    tuple[int, int],  # standalone_noise_range (half-open)
]:
    """Build feature names, role indices, and block ranges for the final matrix.

    We assume that :func:`generate_dataset` has constructed the feature
    matrix ``x`` by horizontal concatenation in the following order::

        x = [x_standalone_informative | x_clusters | x_standalone_noise]

    where

    * ``x_standalone_informative`` contains only standalone informative features,
    * ``x_clusters`` contains, for each :class:`CorrClusterConfig`, the block's
      columns (anchor at ``anchor_index``, the others proxies),
    * ``x_standalone_noise`` contains only standalone noise features.

    Relevance is derived **per column** from each cluster's channels (via the
    shared predicate, exposed as
    :meth:`DatasetConfig.cluster_column_informative_flags`): each cluster column
    goes to ``informative_idx`` or ``noise_idx`` according to whether it carries
    signal, so a single cluster may contribute columns to both. The two index
    lists are nonetheless an exhaustive, disjoint two-way partition.

    Args:
        cfg: Resolved :class:`DatasetConfig` used for generation.
        n_cluster_cols: Number of columns in ``x_clusters``.
        n_inf_cols: Number of columns in ``x_standalone_informative``.
        n_noise_cols: Number of columns in ``x_standalone_noise``.

    Returns:
        names, informative_idx, noise_idx, cluster_indices, anchor_idx,
        standalone_noise_range.
    """
    names: list[str] = []
    informative_idx: list[int] = []
    noise_idx: list[int] = []
    cluster_indices: dict[int, list[int]] = {}
    anchor_idx: dict[int, int] = {}

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

    if n_inf_cols != cfg.n_standalone_informative:
        raise ValueError(
            "generate_informative_features must produce exactly "
            f"cfg.n_standalone_informative={cfg.n_standalone_informative} columns, "
            f"but returned {n_inf_cols}."
        )

    if n_noise_cols != cfg.n_standalone_noise:
        raise ValueError(
            "The noise block must contain exactly cfg.n_standalone_noise "
            f"={cfg.n_standalone_noise} standalone noise features, "
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
    # 1) Standalone informative features: block [0, n_inf_cols)
    # -------------------------------------------------------------
    for j in range(n_inf_cols):
        if cfg.prefixed_feature_naming:
            names.append(f"{cfg.prefix_informative}{j + 1}")
        else:
            names.append(f"feature_{len(names) + 1}")
        informative_idx.append(j)

    # -------------------------------------------------------------
    # 2) Correlated clusters: one contiguous block per CorrClusterConfig.
    # -------------------------------------------------------------
    column_flags = cfg.cluster_column_informative_flags()
    current = n_inf_cols
    for cid, cluster_cfg in enumerate(clusters):
        k = int(cluster_cfg.n_cluster_features)
        cols = list(range(current, current + k))
        cluster_indices[cid] = cols
        anchor_idx[cid] = current + cluster_cfg.anchor_index

        for position, col in enumerate(cols):
            if cfg.prefixed_feature_naming:
                if position == cluster_cfg.anchor_index:
                    names.append(f"{cfg.prefix_corr}{cid + 1}_anchor")
                else:
                    names.append(f"{cfg.prefix_corr}{cid + 1}_{position + 1}")
            else:
                names.append(f"feature_{len(names) + 1}")

            # Derived relevance is per column: a cluster may split across roles.
            if column_flags[cid][position]:
                informative_idx.append(col)
            else:
                noise_idx.append(col)

        current += k

    # -------------------------------------------------------------
    # 3) Standalone noise: block at the end
    # -------------------------------------------------------------
    noise_start = n_inf_cols + n_cluster_cols
    standalone_noise_range = (noise_start, noise_start + n_noise_cols)
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

    # informative_idx and noise_idx must be an exhaustive, disjoint partition.
    if sorted(informative_idx + noise_idx) != list(range(total_cols)):
        raise AssertionError(
            "Internal inconsistency in _make_names_and_roles: informative_idx and "
            "noise_idx must partition range(n_features) exactly."
        )

    return (
        names,
        sorted(informative_idx),
        sorted(noise_idx),
        cluster_indices,
        anchor_idx,
        standalone_noise_range,
    )


# =================
# Public generator
# =================
def generate_dataset(
    cfg: DatasetConfig, return_dataframe: bool = True
) -> tuple[pd.DataFrame | np.ndarray, np.ndarray, DatasetMeta]:
    """Generate synthetic biomedical dataset with specified feature structure.

    Creates a classification dataset with configurable informative features, noise,
    correlated feature clusters (e.g., biological pathways), and optional batch effects.

    Args:
        cfg: Configuration object defining the dataset structure. See
            :class:`~biomedical_data_generator.config.DatasetConfig` for details.
        return_dataframe: If ``True``, return features as a :class:`pandas.DataFrame`
            with named columns. If ``False``, return as a NumPy array.

    Returns:
        tuple: A 3-tuple containing:

            - **x** (:class:`pandas.DataFrame` or :class:`numpy.ndarray`):
              Feature matrix of shape ``(n_samples, n_features)``. Each row represents one sample (e.g., patient),
              each column represents one feature (e.g., biomarker, gene expression value). When returned
              as DataFrame, column names depend on ``cfg.prefixed_feature_naming``:
              when ``True`` (default), names use type-based prefixes
              (``cfg.prefix_informative`` for informative features,
              ``cfg.prefix_corr`` for correlated clusters, ``cfg.prefix_noise``
              for noise), yielding names like ``i1, corr1_anchor, n1``. When
              ``False``, names use sequential numbering ``feature_1, feature_2, ...``.
            - **y** (:class:`numpy.ndarray`):
              Class labels of shape ``(n_samples,)`` with integer values
              ``0, 1, ..., n_classes-1``.
            - **meta** (:class:`DatasetMeta`):
              Metadata object containing feature masks (informative, correlated, noise,
              batch-specific), correlation block specifications, batch assignments,
              and complete generation configuration.

    Examples:
        >>> from biomedical_data_generator.config import (
        ...     DatasetConfig, ClassConfig, StandaloneInformativeGroup
        ... )
        >>> data_cfg_1 = DatasetConfig(
        ...     standalone_informative_groups=[
        ...         StandaloneInformativeGroup(n_features=5, class_sep=1.0)
        ...     ],
        ...     n_standalone_noise=10,
        ...     class_configs=[ClassConfig(n_samples=100, label="healthy"),
        ...                    ClassConfig(n_samples=100, label="diseased")],
        ...     random_state=42
        ... )
        >>> x1, y1, meta_data1 = generate_dataset(data_cfg_1)
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
    # Only standalone noise features are produced here; cluster members (anchors
    # and proxies) already live inside the correlated-cluster block.
    x_noise = sample_distribution(
        distribution=cfg.noise_distribution,
        params=cfg.noise_distribution_params,
        rng=rng_global,
        size=(cfg.n_samples, cfg.n_standalone_noise),
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
    batch_meta: BatchMeta | None = None
    if cfg.batch_effects is not None and cfg.batch_effects.n_batches > 1:
        x, batch_labels, batch_effects = apply_batch_effects_from_config(
            x=x,
            y=y,
            batch_config=cfg.batch_effects,
            rng=rng_global,
        )
        bcfg = cfg.batch_effects
        affected_indices = None if bcfg.affected_features == "all" else list(bcfg.affected_features)
        proportions = tuple(bcfg.proportions) if bcfg.proportions is not None else None
        batch_meta = BatchMeta(
            batch_assignments=batch_labels,
            batch_effects=batch_effects,
            effect_type=bcfg.effect_type,
            effect_strength=bcfg.effect_strength,
            effect_granularity=bcfg.effect_granularity,
            confounding_with_class=bcfg.confounding_with_class,
            proportions=proportions,
            affected_feature_indices=affected_indices,
        )

    # ================================================================
    # STEP 6: Build names and role indices (knows final structure)
    # ================================================================
    (
        names,
        inf_idx,
        noi_idx,
        cluster_idx,
        anch_idx,
        standalone_noise_range,
    ) = _make_names_and_roles(
        cfg,
        n_cluster_cols=x_clusters.shape[1],
        n_inf_cols=x_informative.shape[1],
        n_noise_cols=x_noise.shape[1],
    )

    # Per-group records for the standalone-informative block: column layout and
    # centered per-class offsets resolved straight from the config (no matrix).
    standalone_informative_groups = tuple(
        StandaloneGroupMeta(column_indices=column_indices, per_class_offset=per_class_offset)
        for column_indices, per_class_offset in resolve_standalone_groups(cfg)
    )

    # ================================================================
    # STEP 7: Build metadata
    # ================================================================
    counts = np.bincount(y, minlength=cfg.n_classes)
    # Per-boundary class separation is now declared per standalone-informative
    # group; the legacy scalar field reports the first group's resolved boundaries
    # (recoverable as successive differences of its centered offsets), or zeros
    # when no standalone-informative group is present.
    if standalone_informative_groups:
        class_sep = [float(d) for d in np.diff(standalone_informative_groups[0].per_class_offset)]
    else:
        class_sep = [0.0] * (cfg.n_classes - 1)
    meta = DatasetMeta(
        feature_names=names,
        informative_idx=inf_idx,
        noise_idx=noi_idx,
        corr_cluster_indices=cluster_idx,
        anchor_idx=anch_idx,
        standalone_informative_groups=standalone_informative_groups,
        standalone_noise_range=standalone_noise_range,
        mean_per_class_effect=cluster_meta["mean_per_class_effect"],
        covariance_per_class_correlation=cluster_meta["covariance_per_class_correlation"],
        baseline_correlation=cluster_meta["baseline_correlation"],
        cluster_label=cluster_meta["label"],
        cluster_structure=cluster_meta["structure"],
        cluster_proxy_attenuation=cluster_meta["proxy_attenuation"],
        n_classes=cfg.n_classes,
        class_names=cfg.class_labels,
        samples_per_class={int(k): int(counts[k]) for k in range(cfg.n_classes)},
        class_sep=class_sep,
        batch=batch_meta,
        random_state=cfg.random_state,
        resolved_config=cfg.model_dump(),
    )
    if return_dataframe:
        return pd.DataFrame(x, columns=names), y, meta
    return x, y, meta
