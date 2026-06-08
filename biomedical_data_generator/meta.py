# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Metadata about the generated dataset."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

__all__ = ["BatchMeta", "DatasetMeta", "FeatureRoles", "compute_feature_roles"]


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
    * **Cluster membership** -- a *free* feature is independent and belongs to
      no cluster. Within a correlated cluster, the lead column is the *anchor*
      (the only column shifted directly), and every other member is a *proxy*
      that inherits an attenuated version of the anchor's behaviour through
      correlation.

    Combining the two distinctions yields the six roles, one per index attribute
    below.

    Attributes:
        free_informative_indices:
            List of column indices for free informative features. These are
            independent informative features that are not part of any
            correlated cluster and therefore carry a class-separating mean
            shift on their own.
        informative_anchor_indices:
            List of column indices of anchors of clusters **derived** to carry
            class-discriminative signal (through the mean or the covariance
            channel), independent of the declared ``anchor_role``. Such anchors
            seed the within-cluster correlation shared by their proxies.
        informative_proxy_indices:
            List of column indices of proxy members (non-anchor members) in
            clusters derived informative. Proxies do not receive a direct mean
            shift but inherit an attenuated signal through their correlation
            with the anchor. The degree of attenuation follows the cluster's
            correlation structure — roughly uniform for equicorrelated clusters
            and decaying with distance from the anchor for Toeplitz clusters.
        free_noise_indices:
            List of column indices for free noise features. These are
            independent noise features outside any cluster and carry no
            class-discriminating signal.
        noise_anchor_indices:
            List of column indices of anchors of clusters derived to carry
            **no** class-discriminative signal (neither a class-dependent mean
            shift nor a class-dependent within-cluster correlation), independent
            of the declared ``anchor_role``. They seed a within-cluster
            correlation that is identical across classes.
        noise_proxy_indices:
            List of column indices of proxy members (non-anchor members) in
            clusters derived noise. They are correlated with their anchor and
            form purely structural, signal-free correlated blocks.
        cluster_membership:
            Mapping from ``column_index`` to ``cluster_id`` for every column
            that belongs to a correlated cluster.
    """

    free_informative_indices: list[int]
    informative_anchor_indices: list[int]
    informative_proxy_indices: list[int]
    free_noise_indices: list[int]
    noise_anchor_indices: list[int]
    noise_proxy_indices: list[int]
    cluster_membership: dict[int, int]


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

    # Index sets (0-based column indices)
    informative_idx: list[int]  # includes cluster anchors + free informative features
    noise_idx: list[int]  # independent / free noise features (no anchors)

    # Correlated clusters
    corr_cluster_indices: dict[int, list[int]]  # cluster_id -> list of column indices
    anchor_idx: dict[int, int]  # cluster_id -> anchor col (always the first column of the block)

    # Per-cluster properties (mirroring CorrClusterConfig)
    anchor_role: dict[int, Literal["informative", "noise"]]
    anchor_effect_size: dict[int, float]  # numeric effect size used for the anchor
    anchor_class: dict[int, int | None]  # class index the anchor predicts (one-vs-rest)
    cluster_label: dict[int, str | None]  # descriptive label per cluster (didactic tag)
    cluster_structure: dict[int, Literal["equicorrelated", "toeplitz"]]  # cluster_id -> correlation structure
    cluster_correlation: dict[int, float | dict[int, float]]
    # cluster_id -> within-cluster correlation (global float or
    #   per-class mapping {class_index: correlation})

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


def _mapping_varies_across_classes(per_class_values, n_classes, tol=1e-9):
    """Return whether a per-class mapping differs across classes.

    Absent classes resolve to the 0.0 baseline. A scalar (non-mapping) input
    never varies and must be guarded by the caller.

    Args:
        per_class_values: Mapping from class index to value.
        n_classes: Number of classes to resolve over.
        tol: Numerical tolerance for treating values as equal.

    Returns:
        True if the resolved per-class values are not all equal.
    """
    resolved_values = [float(per_class_values.get(class_index, 0.0)) for class_index in range(n_classes)]
    return (max(resolved_values) - min(resolved_values)) > tol


def _mean_varies_across_classes(anchor_role, anchor_class, effect_size):
    """Return whether the anchor's mean location differs across classes.

    An informative anchor targeting a specific class shifts only that class and
    is discriminative; an anchor with no specific target class shifts every
    class equally and is not.

    Args:
        anchor_role: Declared anchor role.
        anchor_class: Target class index, or None for an untargeted shift.
        effect_size: Resolved numeric effect size (e.g. from
            ``CorrClusterConfig.resolve_anchor_effect_size``).

    Returns:
        True if the anchor introduces a between-class mean difference.
    """
    return anchor_role == "informative" and effect_size > 0.0 and anchor_class is not None


def _cluster_is_informative(anchor_role, anchor_class, effect_size, correlation, n_classes, tol=1e-9):
    """Derive whether a correlated cluster carries class-discriminative signal.

    A cluster is informative if its mean channel varies across classes (first
    moment) OR its within-cluster correlation varies across classes (second
    moment / differential co-expression).

    Args:
        anchor_role: Declared anchor role.
        anchor_class: Target class index, or None.
        effect_size: Resolved numeric anchor effect size (e.g. from
            ``CorrClusterConfig.resolve_anchor_effect_size``).
        correlation: Scalar correlation or a per-class correlation mapping.
        n_classes: Number of classes.
        tol: Numerical tolerance for treating values as equal.

    Returns:
        True if any channel varies across classes.
    """
    mean_signal = _mean_varies_across_classes(anchor_role, anchor_class, effect_size)
    covariance_signal = isinstance(correlation, dict) and _mapping_varies_across_classes(correlation, n_classes, tol)
    return mean_signal or covariance_signal


def compute_feature_roles(meta: DatasetMeta) -> FeatureRoles:
    """Derive the six-way generative feature-role partition from a DatasetMeta.

    The partition is reconstructed purely from the structural index sets that
    the generator already records on ``meta`` (informative and noise indices,
    cluster layout, anchor columns, and per-cluster anchor properties). Each
    cluster's relevance is **derived** from the generated signal — a class-
    dependent mean shift or a class-dependent within-cluster correlation
    (differential co-expression) — rather than read from the declared
    ``anchor_role``. No feature matrix is required.

    Args:
        meta: Resolved dataset metadata produced by
            :func:`biomedical_data_generator.generate_dataset`.

    Returns:
        A :class:`FeatureRoles` instance assigning every feature column to
        exactly one of the six generative roles, together with a
        column-to-cluster membership map.
    """
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
            anchor_role=meta.anchor_role[cluster_id],
            anchor_class=meta.anchor_class[cluster_id],
            effect_size=meta.anchor_effect_size[cluster_id],
            correlation=meta.cluster_correlation[cluster_id],
            n_classes=meta.n_classes,
        ):
            informative_anchor_indices.append(anchor_column)
            informative_proxy_indices.extend(proxy_columns)
        else:
            noise_anchor_indices.append(anchor_column)
            noise_proxy_indices.extend(proxy_columns)

    # meta.informative_idx contains free informative features plus informative
    # anchors; subtract the anchors to recover the free informative features.
    informative_anchor_set = set(informative_anchor_indices)
    free_informative_indices = [idx for idx in meta.informative_idx if idx not in informative_anchor_set]

    # meta.noise_idx already excludes cluster anchors (free noise features only).
    free_noise_indices = list(meta.noise_idx)

    return FeatureRoles(
        free_informative_indices=sorted(free_informative_indices),
        informative_anchor_indices=sorted(informative_anchor_indices),
        informative_proxy_indices=sorted(informative_proxy_indices),
        free_noise_indices=sorted(free_noise_indices),
        noise_anchor_indices=sorted(noise_anchor_indices),
        noise_proxy_indices=sorted(noise_proxy_indices),
        cluster_membership=dict(sorted(cluster_membership.items())),
    )
