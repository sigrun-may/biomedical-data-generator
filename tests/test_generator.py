# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tests for the main dataset generator."""

import numpy as np
import pandas as pd
import pytest

from biomedical_data_generator import (
    BatchEffectsConfig,
    ClassConfig,
    CorrClusterConfig,
    CovarianceChannel,
    DatasetConfig,
    MeanChannel,
    StandaloneInformativeGroup,
    compute_feature_roles,
)
from biomedical_data_generator.generator import _make_names_and_roles, generate_dataset


def test_generate_dataset_basic():
    """Test basic dataset generation."""
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=3, class_sep=1.0)],
        n_standalone_noise=2,
        class_configs=[
            ClassConfig(n_samples=50),
            ClassConfig(n_samples=50),
        ],
        random_state=42,
    )

    X, y, meta = generate_dataset(cfg, return_dataframe=False)

    assert X.shape == (100, 5)
    assert len(y) == 100
    assert meta.n_classes == 2


def test_generate_dataset_returns_dataframe():
    """Test that return_dataframe=True returns DataFrame."""
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=3, class_sep=1.0)],
        n_standalone_noise=2,
        class_configs=[
            ClassConfig(n_samples=50),
            ClassConfig(n_samples=50),
        ],
        random_state=42,
    )

    X, y, meta = generate_dataset(cfg, return_dataframe=True)

    assert isinstance(X, pd.DataFrame)
    assert X.shape == (100, 5)
    assert list(X.columns) == meta.feature_names


def test_generate_dataset_with_clusters():
    """Test dataset generation with correlated clusters."""
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=2, class_sep=1.0)],
        n_standalone_noise=1,
        corr_clusters=[
            CorrClusterConfig(n_cluster_features=4, baseline_correlation=0.7),
        ],
        class_configs=[
            ClassConfig(n_samples=50),
            ClassConfig(n_samples=50),
        ],
        random_state=42,
    )

    X, y, meta = generate_dataset(cfg, return_dataframe=False)

    # No budget any more: 2 standalone informative + 4 cluster members + 1 standalone noise = 7.
    assert X.shape[1] == 7
    assert len(meta.corr_cluster_indices) == 1


def test_generate_dataset_with_batch_effects():
    """Test dataset generation with batch effects."""
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=5, class_sep=1.0)],
        n_standalone_noise=3,
        class_configs=[
            ClassConfig(n_samples=50),
            ClassConfig(n_samples=50),
        ],
        batch_effects=BatchEffectsConfig(n_batches=2),
        random_state=42,
    )

    X, y, meta = generate_dataset(cfg, return_dataframe=False)

    assert X.shape == (100, 8)
    assert meta.batch is not None
    assert len(meta.batch.batch_assignments) == 100
    assert meta.batch.batch_effects is not None


def test_generate_dataset_no_batch_effects_when_n_batches_1():
    """Test that n_batches=1 doesn't apply batch effects."""
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=3, class_sep=1.0)],
        n_standalone_noise=2,
        class_configs=[
            ClassConfig(n_samples=50),
            ClassConfig(n_samples=50),
        ],
        batch_effects=BatchEffectsConfig(n_batches=1),
        random_state=42,
    )

    X, y, meta = generate_dataset(cfg, return_dataframe=False)

    # With n_batches=1, batch effects should not be applied
    assert meta.batch is None


def test_generate_dataset_feature_naming_prefixed():
    """Test prefixed feature naming."""
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=2, class_sep=1.0)],
        n_standalone_noise=1,
        class_configs=[
            ClassConfig(n_samples=50),
            ClassConfig(n_samples=50),
        ],
        prefixed_feature_naming=True,
        prefix_informative="inf_",
        prefix_noise="noise_",
        random_state=42,
    )

    X, y, meta = generate_dataset(cfg, return_dataframe=True)

    # Check that feature names have correct prefixes
    assert any(name.startswith("inf_") for name in meta.feature_names)
    assert any(name.startswith("noise_") for name in meta.feature_names)


def test_generate_dataset_feature_naming_sequential():
    """Test sequential feature naming (not generic)."""
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=2, class_sep=1.0)],
        n_standalone_noise=1,
        class_configs=[
            ClassConfig(n_samples=50),
            ClassConfig(n_samples=50),
        ],
        # Use default or prefixed, not "generic" which may not be valid
        random_state=42,
    )

    X, y, meta = generate_dataset(cfg, return_dataframe=True)

    # Features should have names
    assert len(meta.feature_names) == 3


def test_generate_dataset_multiclass():
    """Test dataset generation with multiple classes."""
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=3, class_sep=[1.5, 2.0])],
        n_standalone_noise=2,
        class_configs=[
            ClassConfig(n_samples=30),
            ClassConfig(n_samples=40),
            ClassConfig(n_samples=30),
        ],
        random_state=42,
    )

    X, y, meta = generate_dataset(cfg, return_dataframe=False)

    assert X.shape == (100, 5)
    assert meta.n_classes == 3
    assert np.sum(y == 0) == 30
    assert np.sum(y == 1) == 40
    assert np.sum(y == 2) == 30


def test_generate_dataset_reproducibility():
    """Test that same random_state produces same results."""
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=3, class_sep=1.0)],
        n_standalone_noise=2,
        class_configs=[
            ClassConfig(n_samples=50),
            ClassConfig(n_samples=50),
        ],
        random_state=123,
    )

    X1, y1, _ = generate_dataset(cfg, return_dataframe=False)
    X2, y2, _ = generate_dataset(cfg, return_dataframe=False)

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_make_names_and_roles_mismatch_cluster_cols_raises():
    """Test that mismatch in cluster columns raises ValueError."""
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=2, class_sep=1.0)],
        n_standalone_noise=1,
        corr_clusters=[
            CorrClusterConfig(n_cluster_features=4, baseline_correlation=0.7),
        ],
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],  # Need 2 classes
    )

    # Pass wrong n_cluster_cols; the cluster-column check fires first.
    with pytest.raises(ValueError, match="Mismatch between x_clusters.shape"):
        _make_names_and_roles(
            cfg,
            n_cluster_cols=5,  # Should be 4
            n_inf_cols=2,
            n_noise_cols=1,
        )


def test_make_names_and_roles_mismatch_inf_cols_raises():
    """Test that mismatch in informative columns raises ValueError."""
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=2, class_sep=1.0)],
        n_standalone_noise=1,
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],  # Need 2 classes
    )

    with pytest.raises(ValueError, match="generate_informative_features must produce"):
        _make_names_and_roles(
            cfg,
            n_cluster_cols=0,
            n_inf_cols=3,  # Should be 2 (n_standalone_informative)
            n_noise_cols=1,
        )


def test_make_names_and_roles_mismatch_noise_cols_raises():
    """Test that mismatch in noise columns raises ValueError."""
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=2, class_sep=1.0)],
        n_standalone_noise=1,
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],  # Need 2 classes
    )

    with pytest.raises(ValueError, match="The noise block must contain"):
        _make_names_and_roles(
            cfg,
            n_cluster_cols=0,
            n_inf_cols=2,
            n_noise_cols=2,  # Should be 1
        )


def test_make_names_and_roles_total_mismatch_raises():
    """Test that we can call _make_names_and_roles successfully."""
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=2, class_sep=1.0)],
        n_standalone_noise=1,
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],
    )

    # Call with correct parameters - should succeed (now returns 6 values).
    names, inf_idx, noi_idx, cluster_idx, anchor_idx, standalone_noise_range = _make_names_and_roles(
        cfg,
        n_cluster_cols=0,
        n_inf_cols=2,
        n_noise_cols=1,
    )

    assert len(names) == 3
    assert standalone_noise_range == (2, 3)


def test_make_names_and_roles_with_channel_free_cluster_is_noise():
    """A channel-free cluster is derived noise: all its columns join noise_idx.

    Relevance is derived from the generated signal. A cluster that carries only a
    structural ``baseline_correlation`` (no mean or covariance channel) has no
    class-discriminative signal, so every one of its columns is assigned to the
    noise partition rather than the informative one.
    """
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=1, class_sep=1.0)],
        n_standalone_noise=1,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                baseline_correlation=0.7,
            ),
        ],
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],  # Need 2 classes
        prefixed_feature_naming=True,
    )

    names, inf_idx, noi_idx, cluster_idx, anchor_idx, standalone_noise_range = _make_names_and_roles(
        cfg,
        n_cluster_cols=3,
        n_inf_cols=1,
        n_noise_cols=1,
    )

    # Layout: [standalone informative (0) | cluster (1,2,3) | standalone noise (4)].
    # The channel-free cluster is derived noise, so its columns join noise_idx.
    assert inf_idx == [0]
    assert noi_idx == [1, 2, 3, 4]
    assert anchor_idx[0] == 1
    assert anchor_idx[0] not in inf_idx
    # Should have one cluster
    assert len(cluster_idx) == 1


def test_make_names_and_roles_with_noise_anchor():
    """Naming and roles when a cluster carries no signal (derived noise)."""
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=2, class_sep=1.0)],
        n_standalone_noise=0,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                baseline_correlation=0.7,
            ),
        ],
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],  # Need 2 classes
        prefixed_feature_naming=True,
    )

    names, inf_idx, noi_idx, cluster_idx, anchor_idx, standalone_noise_range = _make_names_and_roles(
        cfg,
        n_cluster_cols=3,
        n_inf_cols=2,  # 2 standalone informative features
        n_noise_cols=0,  # no standalone noise
    )

    # Layout: [standalone informative (0,1) | cluster (2,3,4)].
    # Informative holds the two standalone features; the derived-noise cluster's
    # columns all land in noise_idx.
    assert inf_idx == [0, 1]
    assert noi_idx == [2, 3, 4]


def test_generate_dataset_metadata_complete():
    """Test that metadata contains all expected fields."""
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=3, class_sep=1.0)],
        n_standalone_noise=2,
        corr_clusters=[
            CorrClusterConfig(n_cluster_features=4, baseline_correlation=0.7),
        ],
        class_configs=[
            ClassConfig(n_samples=50, label="class_A"),
            ClassConfig(n_samples=50, label="class_B"),
        ],
        random_state=42,
    )

    X, y, meta = generate_dataset(cfg, return_dataframe=False)

    # Check all metadata fields
    assert meta.feature_names is not None
    assert meta.informative_idx is not None
    assert meta.noise_idx is not None
    assert meta.corr_cluster_indices is not None
    assert meta.anchor_idx is not None
    assert meta.n_classes == 2
    assert meta.class_names == ["class_A", "class_B"]
    assert meta.samples_per_class == {0: 50, 1: 50}
    assert meta.random_state == 42
    # Per-cluster correlation structure and the channel primitives are exposed first-class.
    assert meta.cluster_structure == {0: "equicorrelated"}
    assert meta.baseline_correlation == {0: 0.7}
    # Channel-free cluster -> no covariance channel mapping recorded.
    assert meta.covariance_per_class_correlation == {0: None}


def test_metadata_exposes_per_cluster_structure_and_correlation():
    """cluster_structure and baseline_correlation reflect each cluster's config."""
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=2, class_sep=1.0)],
        n_standalone_noise=1,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                baseline_correlation=0.7,
                correlation_structure="equicorrelated",
                mean_channel=MeanChannel(per_class_effect={1: 1.0}),  # derived informative
            ),
            CorrClusterConfig(
                n_cluster_features=3,
                baseline_correlation=0.6,
                correlation_structure="toeplitz",  # derived noise (no channels)
            ),
        ],
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],
        random_state=42,
    )

    _, _, meta = generate_dataset(cfg, return_dataframe=False)

    # One entry per cluster_id, reflecting the configured values.
    assert set(meta.cluster_structure) == {0, 1}
    assert set(meta.baseline_correlation) == {0, 1}
    assert meta.cluster_structure == {0: "equicorrelated", 1: "toeplitz"}
    assert meta.baseline_correlation == {0: 0.7, 1: 0.6}


def test_metadata_preserves_class_specific_correlation():
    """A per-class covariance channel mapping is preserved unchanged in the metadata."""
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=1, class_sep=1.0)],
        n_standalone_noise=0,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=4,
                baseline_correlation=0.0,
                correlation_structure="equicorrelated",
                covariance_channel=CovarianceChannel(per_class_correlation={0: 0.0, 1: 0.8}),
                mean_channel=MeanChannel(per_class_effect={1: 1.0}),
            ),
        ],
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],
        random_state=42,
    )

    _, _, meta = generate_dataset(cfg, return_dataframe=False)

    assert meta.cluster_structure == {0: "equicorrelated"}
    assert meta.covariance_per_class_correlation == {0: {0: 0.0, 1: 0.8}}


def test_generate_dataset_with_no_informative():
    """Test generation with only noise features."""
    cfg = DatasetConfig(
        standalone_informative_groups=[],
        n_standalone_noise=5,
        class_configs=[
            ClassConfig(n_samples=50),
            ClassConfig(n_samples=50),
        ],
        random_state=42,
    )

    X, y, meta = generate_dataset(cfg, return_dataframe=False)

    assert X.shape == (100, 5)
    assert len(meta.informative_idx) == 0
    assert len(meta.noise_idx) == 5


def test_generate_dataset_with_no_noise():
    """Test generation with only informative features."""
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=5, class_sep=1.0)],
        n_standalone_noise=0,
        class_configs=[
            ClassConfig(n_samples=50),
            ClassConfig(n_samples=50),
        ],
        random_state=42,
    )

    X, y, meta = generate_dataset(cfg, return_dataframe=False)

    assert X.shape == (100, 5)
    assert len(meta.informative_idx) == 5
    assert len(meta.noise_idx) == 0


def test_generate_dataset_with_noise_anchor_cluster():
    """End-to-end generation works when a cluster carries no signal (derived noise).

    The standalone noise block contains only standalone noise features; the
    derived-noise cluster's members live inside the correlated-cluster block but
    are still part of the noise partition. The total column count equals
    cfg.n_features.
    """
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=1, class_sep=1.0)],
        n_standalone_noise=1,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                baseline_correlation=0.7,
                mean_channel=MeanChannel(per_class_effect={1: 1.0}),  # derived informative
            ),
            CorrClusterConfig(n_cluster_features=3, baseline_correlation=0.5),  # derived noise
        ],
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],
        random_state=42,
    )

    X, y, meta = generate_dataset(cfg, return_dataframe=False)

    assert X.shape == (100, cfg.n_features) == (100, 8)
    # Layout: [standalone inf (0) | informative cluster (1,2,3) | noise cluster (4,5,6) | standalone noise (7)].
    assert meta.informative_idx == [0, 1, 2, 3]
    assert meta.noise_idx == [4, 5, 6, 7]
    # Derived roles: anchor of the informative cluster vs. the noise cluster.
    roles = compute_feature_roles(meta)
    assert roles.informative_anchor_indices == [1]
    assert roles.noise_anchor_indices == [4]
    assert meta.anchor_idx == {0: 1, 1: 4}
    assert meta.corr_cluster_indices == {0: [1, 2, 3], 1: [4, 5, 6]}


def test_generate_dataset_populates_batch_meta():
    """generate_dataset stores all batch settings in meta.batch."""
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=5, class_sep=1.0)],
        n_standalone_noise=3,
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],
        batch_effects=BatchEffectsConfig(
            n_batches=3,
            effect_type="multiplicative",
            effect_strength=0.4,
            effect_granularity="scalar",
            confounding_with_class=0.5,
            affected_features="all",
        ),
        random_state=42,
    )

    _, _, meta = generate_dataset(cfg, return_dataframe=False)

    assert meta.batch is not None
    assert meta.batch.effect_type == "multiplicative"
    assert meta.batch.effect_strength == 0.4
    assert meta.batch.effect_granularity == "scalar"
    assert meta.batch.confounding_with_class == 0.5
    assert meta.batch.affected_feature_indices is None  # "all" maps to None
    assert meta.batch.batch_assignments.shape == (100,)


def test_generate_dataset_batch_meta_affected_feature_indices_list():
    """An explicit affected_features list is stored verbatim in meta.batch."""
    affected = [0, 2, 4]
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=5, class_sep=1.0)],
        n_standalone_noise=3,
        class_configs=[ClassConfig(n_samples=40), ClassConfig(n_samples=40)],
        batch_effects=BatchEffectsConfig(
            n_batches=2,
            effect_strength=0.5,
            affected_features=affected,
        ),
        random_state=0,
    )

    _, _, meta = generate_dataset(cfg, return_dataframe=False)

    assert meta.batch is not None
    assert meta.batch.affected_feature_indices == affected


def test_dataset_meta_to_dict_is_json_serializable_with_batch_effects():
    """meta.to_dict() must be JSON-serializable when batch effects are active."""
    import json

    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=4, class_sep=1.0)],
        n_standalone_noise=2,
        class_configs=[ClassConfig(n_samples=30), ClassConfig(n_samples=30)],
        batch_effects=BatchEffectsConfig(n_batches=2, effect_strength=0.5),
        random_state=7,
    )

    _, _, meta = generate_dataset(cfg, return_dataframe=False)
    payload = meta.to_dict()

    serialized = json.dumps(payload)  # must not raise
    assert isinstance(serialized, str)
    assert isinstance(payload["batch"]["batch_assignments"], list)


def test_dataset_meta_batch_labels_property_matches_batch_meta():
    """The backward-compatible batch_labels property mirrors meta.batch.batch_assignments."""
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=3, class_sep=1.0)],
        n_standalone_noise=2,
        class_configs=[ClassConfig(n_samples=25), ClassConfig(n_samples=25)],
        batch_effects=BatchEffectsConfig(n_batches=2, effect_strength=0.5),
        random_state=1,
    )

    _, _, meta = generate_dataset(cfg, return_dataframe=False)

    assert meta.batch_labels is not None
    np.testing.assert_array_equal(meta.batch_labels, meta.batch.batch_assignments)


def test_dataset_meta_batch_is_none_without_batch_effects():
    """Without batch effects, meta.batch is None and the compat property returns None."""
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=3, class_sep=1.0)],
        n_standalone_noise=2,
        class_configs=[ClassConfig(n_samples=25), ClassConfig(n_samples=25)],
        random_state=1,
    )

    _, _, meta = generate_dataset(cfg, return_dataframe=False)

    assert meta.batch is None
    assert meta.batch_labels is None
