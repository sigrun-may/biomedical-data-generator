# Copyright (c) 2022 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tests for the dataset configuration validation."""

import pytest


def test_n_features_too_small_raises():
    from biomedical_data_generator import CorrClusterConfig, DatasetConfig

    with pytest.raises(ValueError):
        DatasetConfig(
            n_samples=10,
            n_informative=2,
            n_noise=0,
            corr_clusters=[CorrClusterConfig(n_cluster_features=5, rho=0.7)],
            n_features=3,  # too small: needs 2 + (5-1) = 6
        )


def test_relaxed_autofixes_n_features():
    from biomedical_data_generator import CorrClusterConfig, DatasetConfig

    cfg = DatasetConfig.relaxed(
        n_samples=10,
        n_informative=2,
        n_noise=0,
        corr_clusters=[CorrClusterConfig(n_cluster_features=5, rho=0.7)],
        n_features=3,  # will be raised to 6
    )
    assert cfg.n_features == 6


def test_init_has_no_warnings(recwarn):
    from biomedical_data_generator import DatasetConfig

    DatasetConfig(n_samples=5, n_informative=1, n_noise=0)
    assert not recwarn  # no warnings expected


def test_breakdown_matches_required_n_features():
    from biomedical_data_generator import CorrClusterConfig, DatasetConfig

    cfg = DatasetConfig.relaxed(
        n_samples=1,
        n_informative=3,
        n_noise=2,
        corr_clusters=[
            CorrClusterConfig(n_cluster_features=4, rho=0.6),
            CorrClusterConfig(n_cluster_features=3, rho=0.5),
        ],
    )
    b = cfg.breakdown()
    # proxies = (4-1) + (3-1) = 5
    assert b["proxies_from_clusters"] == 5
    assert b["n_features_expected"] == 3 + 2 + 5 == 10
    assert cfg.n_features == 10


def test_corr_clusters_accept_dicts_and_models():
    from biomedical_data_generator import CorrClusterConfig, DatasetConfig

    cfg = DatasetConfig.relaxed(
        n_samples=1,
        n_informative=2,
        corr_clusters=[
            {"n_cluster_features": 3, "rho": 0.7, "anchor_role": "informative"},
            CorrClusterConfig(n_cluster_features=2, rho=0.5),
        ],
    )
    assert cfg.n_features == 2 + (3 - 1) + (2 - 1) == 5
