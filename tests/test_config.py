# Copyright (c) 2022 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tests for the dataset configuration validation (derived counts, breakdown, and input checks)."""

import pytest

from biomedical_data_generator import ClassConfig, StandaloneInformativeGroup


def _minimal_class_configs(n_classes: int = 2, n_per_class: int = 5) -> list[ClassConfig]:
    """Return a minimal list of class configs for testing."""
    return [ClassConfig(n_samples=n_per_class) for _ in range(n_classes)]


def test_manual_derived_fields_forbidden():
    """User must not set derived counts on DatasetConfig.

    ``n_samples`` / ``n_classes`` / ``n_features`` (and ``n_informative`` /
    ``n_noise``) are derived from class_configs and corr_clusters and must be
    computed automatically.
    """
    from biomedical_data_generator import DatasetConfig

    base_kwargs = dict(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=2, class_sep=1.0)],
        n_standalone_noise=3,
        class_configs=_minimal_class_configs(2, 5),
    )

    for forbidden in ("n_samples", "n_classes", "n_features", "n_informative", "n_noise"):
        kwargs = dict(base_kwargs)
        kwargs[forbidden] = 42
        with pytest.raises(ValueError):
            DatasetConfig(**kwargs)


def test_init_has_no_warnings(recwarn):
    """Constructing a basic DatasetConfig should not emit warnings."""
    from biomedical_data_generator import DatasetConfig

    DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=1, class_sep=1.0)],
        n_standalone_noise=0,
        class_configs=_minimal_class_configs(2, 5),
    )
    assert not recwarn  # no warnings expected


def test_breakdown_matches_n_features():
    """breakdown() and n_features must agree on total feature count.

    Scenario:
        - n_standalone_informative = 3
        - n_standalone_noise = 2
        - Two channel-free clusters (both derived noise):
            * k1 = 4
            * k2 = 3
        => n_cluster_members = 7
        => n_features = 3 + 2 + 7 = 12
    """
    from biomedical_data_generator import CorrClusterConfig, DatasetConfig

    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=3, class_sep=1.0)],
        n_standalone_noise=2,
        corr_clusters=[
            CorrClusterConfig(n_cluster_features=4, baseline_correlation=0.6),
            CorrClusterConfig(n_cluster_features=3, baseline_correlation=0.5),
        ],
        class_configs=_minimal_class_configs(2, 1),
    )

    b = cfg.breakdown()
    assert b["n_cluster_members"] == 4 + 3 == 7
    assert b["n_standalone_informative"] == 3
    assert b["n_standalone_noise"] == 2
    assert b["n_features"] == 3 + 2 + 7 == 12
    # Both clusters are channel-free -> derived noise; only standalone informative counts.
    assert b["n_informative"] == 3
    assert b["n_noise"] == 12 - 3 == 9
    # property must match breakdown
    assert cfg.n_features == 12


def test_corr_clusters_accept_dicts_and_models():
    """corr_clusters can contain both dicts and CorrClusterConfig instances.

    Scenario:
        - n_standalone_informative = 2
        - n_standalone_noise = 0
        - clusters:
            * dict: k=3
            * model: k=2
        => n_cluster_members = 5
        => n_features = 2 + 0 + 5 = 7
    """
    from biomedical_data_generator import CorrClusterConfig, DatasetConfig

    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=2, class_sep=1.0)],
        n_standalone_noise=0,
        corr_clusters=[
            {"n_cluster_features": 3, "baseline_correlation": 0.7},
            CorrClusterConfig(n_cluster_features=2, baseline_correlation=0.5),
        ],
        class_configs=_minimal_class_configs(2, 1),
    )
    assert cfg.n_features == 2 + (3 + 2) == 7


def test_derived_counts_and_labels_and_class_sep_broadcast():
    """n_samples, n_classes, class_counts, class_labels and class_sep must be derived correctly.

    Scenario:
        - 3 classes with 4, 6, 10 samples
        - class_sep given as scalar -> broadcast to length (n_classes - 1) = 2
    """
    from biomedical_data_generator import DatasetConfig

    class_cfgs = [
        {"n_samples": 4},
        {"n_samples": 6},
        {"n_samples": 10},
    ]

    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=3, class_sep=2.0)],
        n_standalone_noise=1,
        class_configs=class_cfgs,
    )

    # Derived sample and class counts
    assert cfg.n_classes == 3
    assert cfg.n_samples == 4 + 6 + 10
    assert cfg.class_counts == {0: 4, 1: 6, 2: 10}

    # Auto-generated labels
    assert cfg.class_labels == ["class_0", "class_1", "class_2"]

    # A scalar per-group class_sep broadcasts to n_classes - 1 entries.
    from biomedical_data_generator.features.informative import _resolve_group_sep

    group = cfg.standalone_informative_groups[0]
    assert _resolve_group_sep(group.class_sep, cfg.n_classes) == [2.0, 2.0]

    # n_features = standalone informative + standalone noise (no clusters here)
    assert cfg.n_features == 3 + 1


def test_class_sep_invalid_length_raises():
    """For a multi-class problem, class_sep length must be n_classes - 1.

    If a sequence is provided with wrong length, validation must fail.
    """
    from biomedical_data_generator import DatasetConfig

    class_cfgs = [
        {"n_samples": 3},
        {"n_samples": 3},
        {"n_samples": 4},
    ]

    # 3 classes -> need 2 separation values; here we only pass one
    with pytest.raises(ValueError):
        DatasetConfig(
            standalone_informative_groups=[StandaloneInformativeGroup(n_features=1, class_sep=[1.0])],
            n_standalone_noise=0,
            class_configs=class_cfgs,
        )


def test_channel_class_index_out_of_range_raises():
    """A channel must only reference valid class indices (0 .. n_classes-1)."""
    from biomedical_data_generator import CorrClusterConfig, DatasetConfig, MeanChannel

    # 2 classes -> valid class indices are 0 and 1 only
    with pytest.raises(ValueError):
        DatasetConfig(
            standalone_informative_groups=[StandaloneInformativeGroup(n_features=2, class_sep=1.0)],
            n_standalone_noise=0,
            corr_clusters=[
                CorrClusterConfig(
                    n_cluster_features=2,
                    baseline_correlation=0.7,
                    mean_channel=MeanChannel(per_class_effect={5: 1.0}),  # class 5 out of range
                )
            ],
            class_configs=_minimal_class_configs(2, 5),
        )


def test_corr_cluster_channel_resolution():
    """Channel resolution helpers return per-class signal and correlation."""
    from biomedical_data_generator import CorrClusterConfig, CovarianceChannel, MeanChannel

    cluster = CorrClusterConfig(
        n_cluster_features=3,
        baseline_correlation=0.5,
        mean_channel=MeanChannel(per_class_effect={1: 1.5}),
        covariance_channel=CovarianceChannel(per_class_correlation={0: 0.2, 1: 0.8}),
    )

    # Mean effect: present for class 1, baseline 0.0 for absent classes.
    assert cluster.mean_effect_for_class(1) == 1.5
    assert cluster.mean_effect_for_class(0) == 0.0

    # Correlation: covariance channel overrides baseline for the listed classes.
    assert cluster.effective_correlation_for_class(0) == 0.2
    assert cluster.effective_correlation_for_class(1) == 0.8
    # An absent class falls back to baseline_correlation.
    assert cluster.effective_correlation_for_class(2) == 0.5


def test_corr_cluster_repr_method():
    """Test CorrClusterConfig repr (pydantic default) contains the class name and fields."""
    from biomedical_data_generator import CorrClusterConfig

    cluster = CorrClusterConfig(
        n_cluster_features=3,
        baseline_correlation=0.7,
        correlation_structure="toeplitz",
        anchor_index=1,
    )
    repr_str = repr(cluster)
    # repr should contain the class name
    assert "CorrClusterConfig" in repr_str
    assert "n_cluster_features=3" in repr_str
