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
        batch_assignments: Array of shape (n_samples,) with per-sample batch ids.
        batch_intercepts: Array of shape (n_batches,) with per-batch intercepts added
            to all features (same shift for all features in a batch).
        effect_type: Type of batch effect applied ("intercept", ...).
        effect_strength: Strength of the batch effect (float).
        confounding_strength: Strength of confounding with the target (float).
        proportions: Proportions of samples per batch (if specified).
        affected_feature_indices: List of feature indices affected by the batch effect
            (if not all features are affected).
    """
    batch_assignments: np.ndarray  # (n_samples,)
    batch_intercepts: np.ndarray  # (n_batches,)
    effect_type: str
    effect_strength: float
    confounding_strength: float
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
    pseudo_idx: list[int]  # corr* proxies + free p*
    noise_idx: list[int]

    # Correlated cluster structure
    corr_cluster_indices: dict[int, list[int]]  # cluster_id -> column indices
    anchor_idx: dict[int, int | None]  # cluster_id -> anchor col (or None)
    anchor_role: dict[int, str]  # "informative" | "pseudo" | "noise"
    anchor_effect_size: dict[int, float]  # effect size (beta) for the anchor
    anchor_target_cls: dict[int, int | None]  # target class for the anchor (one-vs-rest)
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
    anchor_mode: AnchorMode = "equalized"
    spread_non_anchors: bool = True

    random_state: int | None = None
    resolved_config: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Convert to a dictionary (e.g., for JSON serialization)."""
        return asdict(self)


# # =========================
# # Feature block structure
# # =========================
# @dataclass
# class FeatureBlock:
#     """Container for a feature block with metadata.
#
#     A feature block represents a logically grouped set of features with
#     shared properties (role, correlation structure, etc.). Blocks are the building units of the dataset.
#     They are generated independently and later assembled into the final dataset.
#     The generator assembles the final dataset by concatenating blocks.
#
#     Attributes:
#         block_type: Type of block ("cluster", "informative", "pseudo", "noise").
#         data: Feature matrix (n_samples, n_features_in_block).
#         names: Feature names for this block.
#         indices: Global column indices in final dataset.
#         role: Feature role ("informative", "pseudo", "noise", 'cluster').
#         cluster_id: Optional cluster identifier for correlated blocks.
#         anchor_idx: Optional anchor column index (global index in dataset).
#         anchor_contrib: Optional (beta, target_class) for informative anchors.
#     """
#
#     block_type: str  # "cluster", "informative", "pseudo", "noise"
#     data: np.ndarray  # (n_samples, n_features)
#     names: list[str]
#     indices: list[int]  # global column indices
#     role: str  # "informative", "pseudo", "noise"
#     cluster_id: int | None = None
#     anchor_col: int | None = None
#
#     @property
#     def n_features(self) -> int:
#         """Number of features in this block."""
#         return len(self.names)
#
#
# @dataclass
# class FeatureBlock:
#     """Container for a feature block with metadata.
#
#     A feature block represents a logically grouped set of features with
#     shared properties (role, correlation structure, etc.). Blocks are
#     generated independently and later assembled into the final dataset.
#
#     Attributes:
#         data: Feature matrix (n_samples, n_features_in_block).
#         role: Feature role ('informative', 'pseudo', 'noise', 'cluster').
#         indices: Global column indices in final dataset.
#         names: Feature names prefix for this block.
#         cluster_id: Optional cluster identifier for correlated blocks.
#         anchor_idx: Optional anchor column index (global index in dataset).
#         anchor_contrib: Optional (beta, target_class) for informative anchors.
#
#     Examples:
#         >>> # Informative block with 3 features
#         >>> block = FeatureBlock(
#         ...     data=np.random.randn(100, 3),
#         ...     role='informative',
#         ...     indices=[0, 1, 2],
#         ...     names=['i1', 'i2', 'i3']
#         ... )
#     """
#
#     data: NDArray[np.float64]
#     role: Literal["informative", "pseudo", "noise", "cluster"]
#     indices: list[int]
#     names: list[str]
#     cluster_id: int | None = None
#     anchor_idx: int | None = None
#     anchor_contrib: tuple[float, int] | None = None
#
#     @property
#     def n_features(self) -> int:
#         """Number of features in this block."""
#         return self.data.shape[1]
#
#     @property
#     def n_samples(self) -> int:
#         """Number of samples in this block."""
#         return self.data.shape[0]
