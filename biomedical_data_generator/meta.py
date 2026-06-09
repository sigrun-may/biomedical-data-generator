# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Metadata about the generated dataset.

This module also hosts the two derivations that read ground truth off a
:class:`DatasetMeta` rather than off the feature matrix:

* :func:`compute_feature_roles` returns the structural six-way partition of the
  columns (which features are informative vs. noise, standalone vs. cluster
  anchor/proxy).
* :func:`compute_feature_strengths` returns the per-feature signal-strength
  annotation, including the set of active signal *channels* per feature.

Both are grounded in the **same** predicate, ``_cluster_is_informative`` (the
mean channel varies across classes OR the within-cluster correlation varies
across classes). Consequently the two derivations must agree: a feature is
placed in an informative role by :func:`compute_feature_roles` **iff**
:func:`compute_feature_strengths` reports a non-empty ``signal_channels`` tuple
for it, and a noise role iff its channels are empty. They are two views of one
predicate, not independent computations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

__all__ = [
    "BatchMeta",
    "DatasetMeta",
    "FeatureRoles",
    "FeatureStrengths",
    "StandaloneGroupMeta",
    "compute_feature_roles",
    "compute_feature_strengths",
]


# =========================
# Batch effects meta
# =========================
@dataclass(frozen=True)
class BatchMeta:
    """Metadata about the batch overlay applied to a dataset.

    Present on :class:`DatasetMeta` only when batch effects were applied;
    otherwise ``DatasetMeta.batch`` is ``None``.

    Attributes:
        batch_assignments:
            Array of shape (n_samples,) with the batch ID per sample.
        batch_effects:
            Per-batch summary of the applied effects as returned by
            ``apply_batch_effects``. For ``effect_granularity="scalar"`` these
            are the exact per-batch shifts (additive) or factors minus 1.0
            (multiplicative); for ``"per_feature"`` they are the mean across
            affected features per batch.
        effect_type:
            Either ``"additive"`` or ``"multiplicative"``.
        effect_strength:
            Standard deviation controlling the effect magnitude.
        effect_granularity:
            Either ``"per_feature"`` or ``"scalar"``.
        confounding_with_class:
            Degree of batch-class correlation in [0.0, 1.0].
        proportions:
            Target batch proportions, or ``None`` for balanced batches.
        affected_feature_indices:
            Column indices that received batch effects, or ``None`` if all
            features were affected.
    """

    batch_assignments: np.ndarray
    batch_effects: np.ndarray
    effect_type: Literal["additive", "multiplicative"]
    effect_strength: float
    effect_granularity: Literal["per_feature", "scalar"]
    confounding_with_class: float
    proportions: tuple[float, ...] | None = None
    affected_feature_indices: list[int] | None = None


# =========================
# Standalone-informative groups
# =========================
@dataclass(frozen=True)
class StandaloneGroupMeta:
    """Resolved metadata for one standalone-informative group.

    Attributes:
        column_indices: The block columns occupied by this group, in order.
        per_class_offset: The centered per-class mean offset applied to every
            member of this group (length n_classes), as produced from the group's
            ``class_sep``.
    """

    column_indices: tuple[int, ...]
    per_class_offset: tuple[float, ...]


# =========================
# Generative feature roles
# =========================
@dataclass(frozen=True)
class FeatureRoles:
    """Generative feature roles derived from a DatasetMeta.

    A purely structural partition of the feature columns into six
    roles that the generator distinguishes. The six roles arise from two
    orthogonal distinctions:

    * **Signal** -- a feature is *informative* when its cluster carries
      class-discriminative signal through **either** a class-dependent mean
      shift **or** a class-dependent within-cluster correlation (differential
      co-expression); *noise* features carry neither.
    * **Cluster membership** -- a *standalone* feature is independent and belongs
      to no cluster. Within a correlated cluster, the structural anchor column is
      the only column shifted directly, and every other member is a *proxy*
      that inherits an attenuated version of the anchor's behaviour through
      correlation.

    Combining the two distinctions yields the six roles, one per index attribute
    below.

    Attributes:
        standalone_informative_indices:
            List of column indices for standalone informative features. These are
            independent informative features that are not part of any
            correlated cluster and therefore carry a class-separating mean
            shift on their own.
        informative_anchor_indices:
            List of column indices of anchors of clusters **derived** to carry
            class-discriminative signal: the cluster is informative iff its mean
            channel varies across classes or its within-cluster correlation varies
            across classes. Such anchors seed the within-cluster correlation
            shared by their proxies.
        informative_proxy_indices:
            List of column indices of proxy members (non-anchor members) in
            clusters derived informative. Proxies do not receive a direct mean
            shift but inherit an attenuated signal through their correlation
            with the anchor. The degree of attenuation follows the cluster's
            correlation structure — roughly uniform for equicorrelated clusters
            and decaying with distance from the anchor for Toeplitz clusters.
        standalone_noise_indices:
            List of column indices for standalone noise features. These are
            independent noise features outside any cluster and carry no
            class-discriminating signal.
        noise_anchor_indices:
            List of column indices of anchors of clusters derived to carry
            **no** class-discriminative signal: neither the mean channel nor the
            within-cluster correlation varies across classes. They seed a
            within-cluster correlation that is identical across classes.
        noise_proxy_indices:
            List of column indices of proxy members (non-anchor members) in
            clusters derived noise. They are correlated with their anchor and
            form purely structural, signal-free correlated blocks.
        cluster_membership:
            Mapping from ``column_index`` to ``cluster_id`` for every column
            that belongs to a correlated cluster.
    """

    standalone_informative_indices: list[int]
    informative_anchor_indices: list[int]
    informative_proxy_indices: list[int]
    standalone_noise_indices: list[int]
    noise_anchor_indices: list[int]
    noise_proxy_indices: list[int]
    cluster_membership: dict[int, int]


# =========================
# Derived per-feature strengths
# =========================
@dataclass(frozen=True)
class FeatureStrengths:
    """Derived per-feature signal-strength annotation.

    All sequences are length ``n_features`` and ordered by column index. Strengths
    are the generative (configured) effect sizes, not finite-sample estimates.

    Attributes:
        mean_strength: First-moment separation per feature in standardized units
            (see ``compute_feature_strengths`` for the unit caveat). 0.0 for any
            feature with no class-dependent mean signal.
        covariance_strength: Second-moment separation per feature, the range of the
            effective per-class within-cluster correlation. 0.0 for any non-cluster
            feature and any cluster with no class-dependent correlation.
        signal_channels: Per feature, the sorted active channels among
            ``("covariance", "mean")``; empty for noise features.
    """

    mean_strength: tuple[float, ...]
    covariance_strength: tuple[float, ...]
    signal_channels: tuple[tuple[str, ...], ...]


# =========================
# Ground-truth dataset meta
# =========================
@dataclass(frozen=True)
class DatasetMeta:
    """Metadata about the generated dataset.

    This captures the resolved ground-truth structure of the dataset
    (feature roles, cluster layout, anchor properties) plus a snapshot
    of the generator configuration.
    """

    # ---------------- core feature layout ----------------

    # Human-readable column names (same order as in X)
    feature_names: list[str]

    # Index sets (0-based column indices); exhaustive two-way partition.
    informative_idx: list[int]  # standalone informative + all members of derived-informative clusters
    noise_idx: list[int]  # complement: standalone noise + all members of derived-noise clusters

    # Correlated clusters
    corr_cluster_indices: dict[int, list[int]]  # cluster_id -> list of column indices
    anchor_idx: dict[int, int]  # cluster_id -> structural anchor column (block start + anchor_index)

    # Per-group records for the standalone-informative block (one per declared
    # group, in declaration order). Together their column_indices tile the block,
    # so roles derive from structure rather than from subtraction.
    standalone_informative_groups: tuple[StandaloneGroupMeta, ...]

    # Per-block column index range (half-open [start, stop)) for standalone noise.
    standalone_noise_range: tuple[int, int]

    # Per-cluster channel primitives (the signal predicate's inputs; relevance is derived).
    mean_per_class_effect: dict[int, dict[int, float] | None]  # cluster_id -> mean channel mapping or None
    covariance_per_class_correlation: dict[int, dict[int, float] | None]  # cluster_id -> covariance mapping or None
    baseline_correlation: dict[int, float]  # cluster_id -> structural baseline correlation
    cluster_label: dict[int, str | None]  # descriptive label per cluster (didactic tag)
    cluster_structure: dict[int, Literal["equicorrelated", "toeplitz"]]  # cluster_id -> correlation structure
    cluster_proxy_attenuation: dict[int, float]  # cluster_id -> anchor-to-proxy mean-propagation multiplier

    # ---------------- provenance / global settings ----------------

    n_classes: int
    class_names: list[str]
    samples_per_class: dict[int, int]
    class_sep: list[float]  # resolved class separation per boundary

    # ---------------- batch effects (optional) ----------------
    batch: BatchMeta | None = None

    # ---------------- generator config snapshot ----------------

    random_state: int | None = None
    resolved_config: dict[str, object] = field(default_factory=dict)

    @property
    def batch_labels(self) -> np.ndarray | None:
        """Per-sample batch assignments, or None if no batch effects were applied.

        Backward-compatibility accessor delegating to :attr:`batch`.
        """
        return self.batch.batch_assignments if self.batch is not None else None

    @property
    def batch_effects(self) -> np.ndarray | None:
        """Per-batch effect summary, or None if no batch effects were applied.

        Backward-compatibility accessor delegating to :attr:`batch`.
        """
        return self.batch.batch_effects if self.batch is not None else None

    def to_dict(self) -> dict[str, object]:
        """Convert to a plain dictionary (e.g., for JSON serialization).

        NumPy arrays are converted to plain lists so the result can be passed
        directly to ``json.dumps``.
        """

        def _convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            if isinstance(obj, np.generic):
                return obj.item()
            return obj

        return _convert(asdict(self))


def _proxy_mean_offset(
    anchor_per_class_offset,
    distance,
    correlation_structure,
    effective_per_class_correlation,
    proxy_attenuation=1.0,
):
    """Propagate an anchor's per-class mean offset to a proxy at a given distance.

    Mirrors the generator's proxy attenuation so that the derived per-feature
    strength matches the realized data exactly. For ``equicorrelated`` the
    propagation factor is the (per-class) within-cluster correlation; for
    ``toeplitz`` it is that correlation raised to the structural distance.

    The factor reproduces ``build_correlation_matrix(...)[anchor, proxy]`` (see
    ``features/correlated.py``) bit-for-bit: an equicorrelated off-diagonal is
    the bare correlation, and a Toeplitz entry is ``rho ** |anchor - proxy|``,
    which is bit-identical to NumPy's ``rho ** exponents`` array form. The
    returned offset preserves the generator's left-to-right multiplication order
    (``offset * proxy_attenuation * factor``).

    Args:
        anchor_per_class_offset: The anchor's resolved per-class mean offsets.
            Scalar or array-like; NumPy broadcasting carries the shape through.
        distance: Structural distance of the proxy from the anchor (>= 1).
        correlation_structure: ``"equicorrelated"`` or ``"toeplitz"``.
        effective_per_class_correlation: The within-cluster correlation per class
            actually used by generation (covariance channel value if present, else
            ``baseline_correlation``).
        proxy_attenuation: Neutral multiplier on the structurally derived
            propagation. ``1.0`` reproduces the v1 model.

    Returns:
        The proxy's resolved per-class mean offsets.

    Raises:
        ValueError: If ``distance < 1`` or ``correlation_structure`` is unknown.
    """
    if distance < 1:
        raise ValueError(f"Proxy distance must be >= 1, got {distance}.")

    if correlation_structure == "equicorrelated":
        factor = effective_per_class_correlation
    elif correlation_structure == "toeplitz":
        factor = effective_per_class_correlation**distance
    else:
        raise ValueError(f"Unknown correlation structure: {correlation_structure}")

    return anchor_per_class_offset * proxy_attenuation * factor


def _range_across_classes(per_class_values, n_classes, default):
    """Return max minus min of a per-class mapping resolved over all classes.

    Absent classes resolve to ``default``.
    """
    resolved_values = [float(per_class_values.get(class_index, default)) for class_index in range(n_classes)]
    return max(resolved_values) - min(resolved_values)


def _mapping_varies_across_classes(per_class_values, n_classes, default, tol=1e-9):
    """Return whether a per-class mapping differs across classes.

    Absent classes resolve to ``default``.

    Args:
        per_class_values: Mapping from class index to value.
        n_classes: Number of classes to resolve over.
        default: Fallback value for classes absent from the mapping.
        tol: Numerical tolerance for treating values as equal.

    Returns:
        True if the resolved per-class values are not all equal.
    """
    return _range_across_classes(per_class_values, n_classes, default) > tol


def _cluster_is_informative(mean_per_class, covariance_per_class, baseline_correlation, n_classes, tol=1e-9):
    """Derive whether a correlated cluster carries class-discriminative signal.

    Informative iff the mean channel varies across classes (first moment) OR the
    within-cluster correlation varies across classes (second moment).

    Args:
        mean_per_class: Per-class mean-shift mapping, or None if absent.
        covariance_per_class: Per-class within-cluster correlation mapping, or None.
        baseline_correlation: Structural correlation used for classes absent from the
            covariance mapping.
        n_classes: Number of classes.
        tol: Numerical tolerance for treating values as equal.

    Returns:
        True if any channel varies across classes.
    """
    mean_signal = mean_per_class is not None and _mapping_varies_across_classes(mean_per_class, n_classes, 0.0, tol)
    covariance_signal = covariance_per_class is not None and _mapping_varies_across_classes(
        covariance_per_class, n_classes, baseline_correlation, tol
    )
    return mean_signal or covariance_signal


def compute_feature_strengths(meta: DatasetMeta) -> FeatureStrengths:
    """Derive per-feature signal strengths from a DatasetMeta.

    Returns three parallel length-n_features sequences: mean_strength (first-moment
    separation in standardized units), covariance_strength (range of per-class
    within-cluster correlation), and signal_channels (the active channels per
    feature).

    A feature carries a signal iff at least one channel is active. The predicate
    on channels agrees with :func:`compute_feature_roles` — a feature is placed
    in an informative role iff its channels are non-empty, and in a noise role
    iff its channels are empty.

    Args:
        meta: Resolved dataset metadata produced by
            :func:`biomedical_data_generator.generate_dataset`.

    Returns:
        A :class:`FeatureStrengths` instance with per-feature signal annotations.
    """
    n_features = len(meta.feature_names)
    mean_strength_list = []
    covariance_strength_list = []
    signal_channels_list = []
    tol = 1e-9

    for column_idx in range(n_features):
        mean_str = 0.0
        covar_str = 0.0
        channels = []

        if column_idx in meta.informative_idx and column_idx not in [
            c for group in meta.standalone_informative_groups for c in group.column_indices
        ]:
            cluster_id = None
            for cid, members in meta.corr_cluster_indices.items():
                if column_idx in members:
                    cluster_id = cid
                    break

            if cluster_id is not None:
                anchor_col = meta.anchor_idx[cluster_id]
                mean_per_class = meta.mean_per_class_effect[cluster_id]
                covariance_per_class = meta.covariance_per_class_correlation[cluster_id]
                baseline_corr = meta.baseline_correlation[cluster_id]
                structure = meta.cluster_structure[cluster_id]
                proxy_attenuation = meta.cluster_proxy_attenuation[cluster_id]

                if column_idx == anchor_col:
                    anchor_per_class_offset = mean_per_class if mean_per_class is not None else {}
                    mean_str = _range_across_classes(anchor_per_class_offset, meta.n_classes, 0.0)
                    covar_str = _range_across_classes(
                        covariance_per_class if covariance_per_class is not None else {},
                        meta.n_classes,
                        baseline_corr,
                    )
                else:
                    distance = abs(column_idx - anchor_col)
                    anchor_per_class_offset = mean_per_class if mean_per_class is not None else {}
                    proxy_per_class_offset = {}
                    for class_idx in range(meta.n_classes):
                        anchor_offset = float(anchor_per_class_offset.get(class_idx, 0.0))
                        effective_corr = (
                            covariance_per_class.get(class_idx, baseline_corr)
                            if covariance_per_class is not None
                            else baseline_corr
                        )
                        proxy_offset = _proxy_mean_offset(
                            anchor_per_class_offset=anchor_offset,
                            distance=distance,
                            correlation_structure=structure,
                            effective_per_class_correlation=effective_corr,
                            proxy_attenuation=proxy_attenuation,
                        )
                        if proxy_offset != 0.0:
                            proxy_per_class_offset[class_idx] = proxy_offset
                    mean_str = _range_across_classes(proxy_per_class_offset, meta.n_classes, 0.0)
                    covar_str = _range_across_classes(
                        covariance_per_class if covariance_per_class is not None else {},
                        meta.n_classes,
                        baseline_corr,
                    )
        elif column_idx in [c for group in meta.standalone_informative_groups for c in group.column_indices]:
            for group in meta.standalone_informative_groups:
                if column_idx in group.column_indices:
                    mean_str = _range_across_classes(
                        {i: v for i, v in enumerate(group.per_class_offset)},
                        meta.n_classes,
                        0.0,
                    )
                    break

        if mean_str > tol:
            channels.append("mean")
        if covar_str > tol:
            channels.append("covariance")

        mean_strength_list.append(mean_str)
        covariance_strength_list.append(covar_str)
        signal_channels_list.append(tuple(sorted(channels)))

    return FeatureStrengths(
        mean_strength=tuple(mean_strength_list),
        covariance_strength=tuple(covariance_strength_list),
        signal_channels=tuple(signal_channels_list),
    )


def compute_feature_roles(meta: DatasetMeta) -> FeatureRoles:
    """Derive the six-way generative feature-role partition from a DatasetMeta.

    The partition is reconstructed purely from the structural block ranges that
    the generator records on ``meta`` (the per-group standalone-informative
    column indices, the standalone-noise column range, each cluster's columns,
    and its structural anchor column) together with the per-cluster channel mappings. Each cluster's
    relevance is **derived** from the generated signal — a class-dependent mean
    shift or a class-dependent within-cluster correlation (differential
    co-expression) — never read from a declared role. No feature matrix is
    required.

    Args:
        meta: Resolved dataset metadata produced by
            :func:`biomedical_data_generator.generate_dataset`.

    Returns:
        A :class:`FeatureRoles` instance assigning every feature column to
        exactly one of the six generative roles, together with a
        column-to-cluster membership map.
    """
    standalone_informative_indices = [
        column for group in meta.standalone_informative_groups for column in group.column_indices
    ]
    standalone_noise_indices = list(range(*meta.standalone_noise_range))

    informative_anchor_indices: list[int] = []
    informative_proxy_indices: list[int] = []
    noise_anchor_indices: list[int] = []
    noise_proxy_indices: list[int] = []
    cluster_membership: dict[int, int] = {}

    for cluster_id, member_columns in meta.corr_cluster_indices.items():
        anchor_column = meta.anchor_idx[cluster_id]
        proxy_columns = [column for column in member_columns if column != anchor_column]

        for column in member_columns:
            cluster_membership[column] = cluster_id

        if _cluster_is_informative(
            mean_per_class=meta.mean_per_class_effect[cluster_id],
            covariance_per_class=meta.covariance_per_class_correlation[cluster_id],
            baseline_correlation=meta.baseline_correlation[cluster_id],
            n_classes=meta.n_classes,
        ):
            informative_anchor_indices.append(anchor_column)
            informative_proxy_indices.extend(proxy_columns)
        else:
            noise_anchor_indices.append(anchor_column)
            noise_proxy_indices.extend(proxy_columns)

    return FeatureRoles(
        standalone_informative_indices=sorted(standalone_informative_indices),
        informative_anchor_indices=sorted(informative_anchor_indices),
        informative_proxy_indices=sorted(informative_proxy_indices),
        standalone_noise_indices=sorted(standalone_noise_indices),
        noise_anchor_indices=sorted(noise_anchor_indices),
        noise_proxy_indices=sorted(noise_proxy_indices),
        cluster_membership=dict(sorted(cluster_membership.items())),
    )
