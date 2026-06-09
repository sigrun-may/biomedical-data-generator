# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Validation and derived-count tests for the channel-based config surface."""

import pytest
from pydantic import ValidationError

from biomedical_data_generator.config import (
    ClassConfig,
    CorrClusterConfig,
    CovarianceChannel,
    DatasetConfig,
    MeanChannel,
    StandaloneInformativeGroup,
)


def _two_classes():
    return [ClassConfig(n_samples=20), ClassConfig(n_samples=20)]


# ---------------------------------------------------------------------------
# Field-local CorrClusterConfig validation
# ---------------------------------------------------------------------------
def test_n_cluster_features_below_two_raises():
    with pytest.raises(ValidationError):
        CorrClusterConfig(n_cluster_features=1)


def test_anchor_index_out_of_range_raises():
    with pytest.raises(ValidationError, match="anchor_index"):
        CorrClusterConfig(n_cluster_features=3, anchor_index=3)


def test_baseline_correlation_out_of_range_raises():
    with pytest.raises(ValidationError, match="baseline_correlation"):
        CorrClusterConfig(n_cluster_features=3, baseline_correlation=1.5)


def test_channel_correlation_out_of_range_raises():
    with pytest.raises(ValidationError, match="covariance_channel"):
        CorrClusterConfig(
            n_cluster_features=3,
            covariance_channel=CovarianceChannel(per_class_correlation={0: 0.0, 1: 1.4}),
        )


# ---------------------------------------------------------------------------
# Cross-cutting per-class key validation (needs n_classes -> on DatasetConfig)
# ---------------------------------------------------------------------------
def test_mean_channel_out_of_range_class_key_raises():
    with pytest.raises(ValidationError, match="mean_channel"):
        DatasetConfig(
            standalone_informative_groups=[StandaloneInformativeGroup(n_features=1, class_sep=1.0)],
            n_standalone_noise=1,
            class_configs=_two_classes(),
            corr_clusters=[
                CorrClusterConfig(
                    n_cluster_features=3,
                    mean_channel=MeanChannel(per_class_effect={2: 1.0}),
                )
            ],
        )


def test_covariance_channel_out_of_range_class_key_raises():
    with pytest.raises(ValidationError, match="covariance_channel"):
        DatasetConfig(
            standalone_informative_groups=[StandaloneInformativeGroup(n_features=1, class_sep=1.0)],
            n_standalone_noise=1,
            class_configs=_two_classes(),
            corr_clusters=[
                CorrClusterConfig(
                    n_cluster_features=3,
                    baseline_correlation=0.3,
                    covariance_channel=CovarianceChannel(per_class_correlation={0: 0.3, 5: 0.8}),
                )
            ],
        )


# ---------------------------------------------------------------------------
# Derived properties (no declared budget)
# ---------------------------------------------------------------------------
def test_derived_counts_for_representative_config():
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=2, class_sep=1.0)],
        n_standalone_noise=3,
        class_configs=_two_classes(),
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                baseline_correlation=0.6,
                mean_channel=MeanChannel(per_class_effect={1: 1.0}),  # derived informative
            ),
            CorrClusterConfig(n_cluster_features=4, baseline_correlation=0.5),  # derived noise
        ],
    )

    # n_features = standalone (2 + 3) + cluster members (3 + 4) = 12
    assert cfg.n_features == 12
    # n_informative = standalone informative (2) + members of the informative cluster (3) = 5
    assert cfg.n_informative == 5
    # n_noise = complement = 7
    assert cfg.n_noise == 7
    assert cfg.n_informative + cfg.n_noise == cfg.n_features


def test_derived_informative_counts_a_covariance_only_cluster():
    cfg = DatasetConfig(
        standalone_informative_groups=[],
        n_standalone_noise=1,
        class_configs=_two_classes(),
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                baseline_correlation=0.0,
                covariance_channel=CovarianceChannel(per_class_correlation={0: 0.0, 1: 0.8}),
            )
        ],
    )
    # The covariance-only cluster is derived informative -> its 3 members count.
    assert cfg.n_informative == 3
    assert cfg.n_noise == 1
    assert cfg.n_features == 4
