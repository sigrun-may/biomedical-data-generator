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

__all__ = [
    "BatchMeta",
    "DatasetMeta",
    "FeatureRoles",
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
# Generative feature roles
# =========================
@dataclass(frozen=True)
class FeatureRoles:
    """Generative feature roles derived from a DatasetMeta.

    A purely structural partition of the feature columns into six
    roles that the generator distinguishes. The six roles arise from two
    orthogonal distinctions:

    * **Signal** -- *informative* features encode a class mean shift, *noise*
      features do not.
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
            List of column indices of cluster anchors whose ``anchor_role`` is
            ``"informative"``. Anchors receive the class-specific mean shift
            and seed the within-cluster correlation shared by their proxies.
        informative_proxy_indices:
            List of column indices of proxy members in informative clusters
            (non-anchor members). Proxies do not receive a direct mean shift
            but inherit an attenuated signal through their correlation with
            the informative anchor. The degree of attenuation follows the
            cluster's correlation structure — roughly uniform for
            equicorrelated clusters and decaying with distance from the anchor
            for Toeplitz clusters.
        free_noise_indices:
            List of column indices for free noise features. These are
            independent noise features outside any cluster and carry no
            class-discriminating signal.
        noise_anchor_indices:
            List of column indices of cluster anchors whose ``anchor_role`` is
            ``"noise"``. Noise anchors seed a within-cluster correlation but
            do not receive a class-specific mean shift.
        noise_proxy_indices:
            List of column indices of proxy members in noise clusters
            (non-anchor members) that are correlated with their noise anchor
            and form purely structural, signal-free correlated blocks.
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
    corr_between: float  # correlation between different clusters/roles

    # ---------------- batch effects (optional) ----------------
    batch: BatchMeta | None = None

    # ---------------- generator config snapshot ----------------

    random_state: int | None = None
    resolved_config: dict[str, object] = field(default_factory=dict)

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
