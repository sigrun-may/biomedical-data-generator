# Copyright (c) 2022 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tests for the dataset configuration validation (derived counts, breakdown, and input checks)."""

import pytest


def _minimal_class_configs(n_classes: int = 2, n_per_class: int = 5) -> list[dict]:
    """Return a minimal list of raw class configs for testing.

    We pass dicts so that Pydantic constructs ClassConfig internally.
    """
    return [{"n_samples": n_per_class} for _ in range(n_classes)]


def test_manual_derived_fields_forbidden():
    """User must not set n_samples / n_classes / n_features on DatasetConfig.

    These values are derived from class_configs and corr_clusters and
    should be computed automatically.
    """
    from biomedical_data_generator import DatasetConfig

    base_kwargs = dict(
        n_informative=2,
        n_noise=3,
        class_configs=_minimal_class_configs(2, 5),
    )

    for forbidden in ("n_samples", "n_classes", "n_features"):
        kwargs = dict(base_kwargs)
        kwargs[forbidden] = 42
        with pytest.raises(ValueError):
            DatasetConfig(**kwargs)


def test_init_has_no_warnings(recwarn):
    """Constructing a basic DatasetConfig should not emit warnings."""
    from biomedical_data_generator import DatasetConfig

    DatasetConfig(
        n_informative=1,
        n_noise=0,
        class_configs=_minimal_class_configs(2, 5),
    )
    assert not recwarn  # no warnings expected


def test_breakdown_matches_n_features():
    """breakdown() and n_features must agree on total feature count.

    Scenario:
        - n_informative = 3
        - n_noise = 2
        - Two clusters:
            * k1 = 4 -> 3 proxies
            * k2 = 3 -> 2 proxies
        => proxies_from_clusters = 5
        => n_features = 3 + 2 + 5 = 10
    """
    from biomedical_data_generator import CorrClusterConfig, DatasetConfig

    cfg = DatasetConfig(
        n_informative=3,
        n_noise=2,
        corr_clusters=[
            CorrClusterConfig(n_cluster_features=4, rho=0.6),
            CorrClusterConfig(n_cluster_features=3, rho=0.5),
        ],
        class_configs=_minimal_class_configs(2, 1),
    )

    b = cfg.breakdown()
    # proxies = (4-1) + (3-1) = 5
    assert b["proxies_from_clusters"] == 5
    assert b["n_informative_total"] == 3
    assert b["n_noise_total"] == 2
    assert b["n_features"] == 3 + 2 + 5 == 10
    # property must match breakdown
    assert cfg.n_features == 10


def test_corr_clusters_accept_dicts_and_models():
    """corr_clusters can contain both dicts and CorrClusterConfig instances.

    Scenario:
        - n_informative = 2
        - n_noise = 0
        - clusters:
            * dict: k=3 -> 2 proxies
            * model: k=2 -> 1 proxy
        => proxies = 3
        => n_features = 2 + 0 + 3 = 5
    """
    from biomedical_data_generator import CorrClusterConfig, DatasetConfig

    cfg = DatasetConfig(
        n_informative=2,
        n_noise=0,
        corr_clusters=[
            {"n_cluster_features": 3, "rho": 0.7, "anchor_role": "informative"},
            CorrClusterConfig(n_cluster_features=2, rho=0.5),
        ],
        class_configs=_minimal_class_configs(2, 1),
    )
    assert cfg.n_features == 2 + (3 - 1) + (2 - 1) == 5


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
        n_informative=3,
        n_noise=1,
        class_configs=class_cfgs,
        class_sep=2.0,
    )

    # Derived sample and class counts
    assert cfg.n_classes == 3
    assert cfg.n_samples == 4 + 6 + 10
    assert cfg.class_counts == {0: 4, 1: 6, 2: 10}

    # Auto-generated labels
    assert cfg.class_labels == ["class_0", "class_1", "class_2"]

    # class_sep must be broadcast to n_classes - 1 entries
    assert cfg.class_sep == [2.0, 2.0]

    # n_features = n_informative + n_noise (no clusters here)
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
            n_informative=1,
            n_noise=0,
            class_configs=class_cfgs,
            class_sep=[1.0],
        )


def test_anchor_class_out_of_range_raises():
    """anchor_class must be a valid class index (0 .. n_classes-1)."""
    from biomedical_data_generator import CorrClusterConfig, DatasetConfig

    # 2 classes -> valid anchor_class are 0 and 1 only
    with pytest.raises(ValueError):
        DatasetConfig(
            n_informative=2,
            n_noise=0,
            corr_clusters=[
                CorrClusterConfig(
                    n_cluster_features=2,
                    rho=0.7,
                    anchor_role="informative",
                    anchor_class=5,  # out of range
                )
            ],
            class_configs=_minimal_class_configs(2, 5),
        )


def test_corr_between_out_of_range_raises():
    """corr_between must lie within [-1, 1]."""
    from biomedical_data_generator import DatasetConfig

    with pytest.raises(ValueError):
        DatasetConfig(
            n_informative=1,
            n_noise=0,
            corr_between=1.5,  # invalid
            class_configs=_minimal_class_configs(2, 5),
        )
