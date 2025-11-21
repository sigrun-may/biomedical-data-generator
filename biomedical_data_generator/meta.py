# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Metadata about the generated dataset."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from biomedical_data_generator.config import AnchorMode


@dataclass
class BatchMeta:
    """Metadata about applied batch effects.

    Attributes:
        batch_assignments: Array of shape (n_samples,) with per-sample batch IDs.
        batch_intercepts: Mapping from batch_id to array of intercepts per affected feature.
            Structure: {batch_id: np.ndarray of shape (n_affected_features,)}
            For example, with 3 batches and 5 affected features:
            {0: array([0.5, -0.3, 0.8, ...]), 1: array([...]), 2: array([...])}
        effect_type: Type of batch effect applied ("additive" or "multiplicative").
        effect_strength: Standard deviation of batch intercepts (controls magnitude).
        confounding_with_class: Degree of correlation between batch and class (0.0-1.0).
            0.0 = independent, 1.0 = perfect confounding.
        proportions: Proportions of samples per batch (if specified).
        affected_feature_indices: List of feature indices affected by batch effects
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
# Ground-truth meta
# =========================
@dataclass(frozen=True)
class DatasetMeta:
    """Metadata about the generated dataset."""

    feature_names: list[str]

    informative_idx: list[int]  # includes cluster anchors + free i*
    noise_idx: list[int]  # Independent noise

    # Correlated cluster structure
    corr_cluster_indices: dict[int, list[int]]  # cluster_id -> column indices
    anchor_idx: dict[int, int | None]  # cluster_id -> anchor col (or None)
    anchor_role: dict[int, str]  # "informative" | "noise"
    anchor_effect_size: dict[int, float]  # effect size (beta) for the anchor
    anchor_class: dict[int, int | None]  # target class for the anchor (one-vs-rest)
    cluster_label: dict[int, str | None]  # didactic tags per cluster

    # Provenance / signal settings
    n_classes: int
    class_names: list[str]
    samples_per_class: dict[int, int]
    class_sep: list[float]
    corr_between: float

    # --- optional (with defaults) ---
    anchor_mode: AnchorMode = "equalized"

    # Batch effects metadata
    batch_labels: np.ndarray | None = None
    batch_intercepts: dict[int, np.ndarray] | None = None  # <- Same type!
    batch_config: dict[str, object] | None = None

    random_state: int | None = None
    resolved_config: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Convert to a dictionary (e.g., for JSON serialization)."""
        return asdict(self)
