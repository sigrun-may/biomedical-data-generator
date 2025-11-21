# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Metadata about the generated dataset."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


# =========================
# Batch effects meta
# =========================
@dataclass
class BatchMeta:
    """Metadata about applied batch effects.

    Attributes:
        batch_assignments:
            Array of shape (n_samples,) with per-sample batch IDs.

        batch_intercepts:
            Mapping from batch_id to array of intercepts per affected feature.
            Structure: {batch_id: np.ndarray of shape (n_affected_features,)}.
            For example, with 3 batches and 5 affected features:
            {0: array([0.5, -0.3, 0.8, ...]), 1: array([...]), 2: array([...])}

        effect_type:
            Type of batch effect ("additive" or "multiplicative").

        effect_strength:
            Standard deviation of batch intercepts (controls magnitude).

        confounding_with_class:
            Degree of correlation between batch and class (0.0–1.0).
            0.0 = independent, 1.0 = perfect confounding.

        proportions:
            Proportions of samples per batch (if specified).

        affected_feature_indices:
            List of feature indices affected by batch effects
            (None if all features are affected).
    """

    batch_assignments: np.ndarray  # (n_samples,)
    batch_intercepts: dict[int, np.ndarray]  # batch_id -> intercepts per affected feature
    effect_type: str  # "additive" or "multiplicative"
    effect_strength: float
    confounding_with_class: float
    proportions: tuple[float, ...] | None = None
    affected_feature_indices: list[int] | None = None


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
    noise_idx: list[int]        # independent / free noise features (no anchors)

    # Correlated clusters
    corr_cluster_indices: dict[int, list[int]]  # cluster_id -> list of column indices
    anchor_idx: dict[int, int | None]           # cluster_id -> anchor col (or None)

    # Per-cluster properties (mirroring CorrClusterConfig)
    anchor_role: dict[int, str]                 # "informative" | "noise"
    anchor_effect_size: dict[int, float]        # numeric effect size used for the anchor
    anchor_class: dict[int, int | None]         # class index the anchor predicts (one-vs-rest)
    cluster_label: dict[int, str | None]        # descriptive label per cluster (didactic tag)

    # ---------------- provenance / global settings ----------------

    n_classes: int
    class_names: list[str]
    samples_per_class: dict[int, int]
    class_sep: list[float]                      # resolved class separation per boundary
    corr_between: float                         # correlation between different clusters/roles

    # ---------------- batch effects (optional) ----------------

    batch_labels: np.ndarray | None = None                      # shape (n_samples,)
    batch_intercepts: dict[int, np.ndarray] | None = None       # batch_id -> intercepts per affected feature
    batch_config: dict[str, object] | None = None               # serialized BatchEffectsConfig

    # ---------------- generator config snapshot ----------------

    random_state: int | None = None
    resolved_config: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Convert to a plain dictionary (e.g., for JSON serialization)."""
        return asdict(self)
