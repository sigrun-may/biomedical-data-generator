# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Metadata about the generated dataset."""

from dataclasses import asdict, dataclass, field

import numpy as np

from biomedical_data_generator.config import AnchorMode


@dataclass
class BatchMeta:
    """Metadata about batch effects."""

    batch_ids: np.ndarray  # (n_samples,)
    batch_offsets: np.ndarray  # (n_batches,)
    batches_majority_class: np.ndarray | None  # (n_batches,) or None
    scope: str
    sd: float


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
