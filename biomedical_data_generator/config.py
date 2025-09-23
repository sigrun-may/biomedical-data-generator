# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Configuration models for the dataset generator."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Literal

import yaml
from pydantic import BaseModel, Field


class CorrCluster(BaseModel):
    """Configuration for a correlated feature cluster.

    Attributes:
        size: Number of features in the cluster.
        rho: Target correlation within the cluster (0 ≤ rho < 1).
        structure: Correlation structure ("equicorrelated" or "ar1").
        anchor_role: Whether the cluster has an anchor that affects y
            ("informative") or is purely latent ("latent").
        anchor_beta: Effect size for the anchor (only if informative).
        anchor_class: Class index (0..n_classes-1) for which the anchor
            contributes positively. Ignored if anchor_role="latent".
        random_state: Optional seed for reproducibility of this cluster.
        label: Optional didactic label for clarity in teaching contexts.
    """
    size: int
    rho: float = 0.7
    structure: Literal["equicorrelated", "ar1"] = "equicorrelated"
    anchor_role: Literal["informative", "latent"] = "latent"
    anchor_beta: float = 1.0
    anchor_class: Optional[int] = 0
    random_state: Optional[int] = None
    label: Optional[str] = None


class DatasetConfig(BaseModel):
    """Configuration for generating a synthetic classification dataset.

    Attributes:
        n_samples: Number of samples (rows).
        n_features: Total number of features.
        n_informative: Number of informative features (including anchors).
        n_pseudo: Number of pseudo features (correlated proxies or free).
        n_noise: Number of pure noise features.
        class_sep: Global scaling factor for signal strength.
        n_classes: Number of classes (>=2).
        weights: Optional class proportions of length n_classes.
        corr_between: Global coupling between all cluster features.
        corr_clusters: Optional list of correlated feature clusters.
        feature_naming: How to name features ("prefixed" or "generic").
        prefix_informative: Prefix for informative features.
        prefix_corr: Prefix for correlated proxy features.
        prefix_pseudo: Prefix for free pseudo features.
        prefix_noise: Prefix for noise features.
        random_state: Optional global random seed.
    """
    n_samples: int = Field(200, ge=1)
    n_features: int
    n_informative: int
    n_pseudo: int = 0
    n_noise: int = 0
    class_sep: float = 1.0

    # Multiclass extension
    n_classes: int = 2
    weights: Optional[Sequence[float]] = None  # length n_classes or None

    # Correlation
    corr_between: float = 0.0
    corr_clusters: Optional[List[CorrCluster]] = None

    # Naming
    feature_naming: Literal["prefixed", "generic"] = "prefixed"
    prefix_informative: str = "i"
    prefix_corr: str = "corr"
    prefix_pseudo: str = "p"
    prefix_noise: str = "n"

    # Randomness
    random_state: Optional[int] = None


@classmethod
def from_yaml(cls, path: str | Path) -> "DatasetConfig":
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return cls(**data)