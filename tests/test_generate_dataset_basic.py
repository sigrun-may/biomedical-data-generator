# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Basic tests for the dataset generator."""

import numpy as np
import pytest

from biomedical_data_generator import CorrCluster, DatasetConfig
from biomedical_data_generator.generator import generate_dataset


def test_shapes_and_roles_with_cluster_proxies():
    # One cluster size 4 => (4-1)=3 proxies + 2 free informative + 1 noise = 6 total features
    cfg = DatasetConfig(
        n_samples=120,
        n_informative=2,
        n_pseudo=0,
        n_noise=1,
        corr_clusters=[
            CorrCluster(
                n_cluster_features=4,
                rho=0.7,
                structure="equicorrelated",
                anchor_role="informative",
                anchor_effect_size=1.0,
            )
        ],
        n_features=2 + 0 + 1 + (4 - 1),
        class_counts={0: 60, 1: 60},  # Explicit class counts
        random_state=11,
    )
    X, y, meta = generate_dataset(cfg, return_dataframe=False)
    assert X.shape == (120, cfg.n_features)
    assert y.shape == (120,)
    # indices consistency
    idx_all = set(meta.informative_idx) | set(meta.pseudo_idx) | set(meta.noise_idx)
    assert len(idx_all) == cfg.n_features
    # cluster index map present
    assert isinstance(meta.corr_cluster_indices, dict) and len(meta.corr_cluster_indices) >= 1


def test_class_counts_exact_match():
    cfg = DatasetConfig(
        n_samples=400,
        n_informative=3,
        n_pseudo=0,
        n_noise=0,
        corr_clusters=[
            CorrCluster(
                n_cluster_features=3, rho=0.6, structure="toeplitz", anchor_role="informative", anchor_effect_size=1.0
            )
        ],
        n_features=3 + (3 - 1),
        n_classes=3,
        class_counts={0: 80, 1: 200, 2: 120},  # Explicit class counts
        class_sep=1.2,
        random_state=123,
    )
    X, y, meta = generate_dataset(cfg, return_dataframe=False)
    # Check exact match with requested counts
    assert meta.y_counts == {0: 80, 1: 200, 2: 120}
    counts = np.bincount(y, minlength=cfg.n_classes)
    assert counts[0] == 80 and counts[1] == 200 and counts[2] == 120


def test_invalid_n_classes_raises():
    cfg = DatasetConfig(
        n_samples=50,
        n_informative=2,
        n_pseudo=0,
        n_noise=0,
        n_features=2,
        n_classes=1,
        class_counts={0: 50}  # Even with class_counts, n_classes=1 should fail
    )
    with pytest.raises(ValueError):
        generate_dataset(cfg, return_dataframe=False)
