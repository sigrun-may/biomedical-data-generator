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
    DatasetConfig,
)
from biomedical_data_generator.generator import _make_names_and_roles, generate_dataset


def test_generate_dataset_basic():
    """Test basic dataset generation."""
    cfg = DatasetConfig(
        n_informative=3,
        n_noise=2,
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
        n_informative=3,
        n_noise=2,
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
        n_informative=2,
        n_noise=1,
        corr_clusters=[
            CorrClusterConfig(n_cluster_features=4, correlation=0.7),
        ],
        class_configs=[
            ClassConfig(n_samples=50),
            ClassConfig(n_samples=50),
        ],
        random_state=42,
    )

    X, y, meta = generate_dataset(cfg, return_dataframe=False)

    # With informative anchor: 2 free informative + 1 anchor (from cluster)
    # Total = 2 free informative + 4 cluster features + 1 noise
    # But n_informative counts anchors, so it's 2 total informative (includes 1 anchor)
    # which means 1 free informative + 4 cluster features + 1 noise = 6
    assert X.shape[1] == 6
    assert len(meta.corr_cluster_indices) == 1


def test_generate_dataset_with_batch_effects():
    """Test dataset generation with batch effects."""
    cfg = DatasetConfig(
        n_informative=5,
        n_noise=3,
        class_configs=[
            ClassConfig(n_samples=50),
            ClassConfig(n_samples=50),
        ],
        batch_effects=BatchEffectsConfig(n_batches=2),
        random_state=42,
    )

    X, y, meta = generate_dataset(cfg, return_dataframe=False)

    assert X.shape == (100, 8)
    assert meta.batch_labels is not None
    assert len(meta.batch_labels) == 100
    assert meta.batch_effects is not None


def test_generate_dataset_no_batch_effects_when_n_batches_1():
    """Test that n_batches=1 doesn't apply batch effects."""
    cfg = DatasetConfig(
        n_informative=3,
        n_noise=2,
        class_configs=[
            ClassConfig(n_samples=50),
            ClassConfig(n_samples=50),
        ],
        batch_effects=BatchEffectsConfig(n_batches=1),
        random_state=42,
    )

    X, y, meta = generate_dataset(cfg, return_dataframe=False)

    # With n_batches=1, batch effects should not be applied
    assert meta.batch_labels is None


def test_generate_dataset_feature_naming_prefixed():
    """Test prefixed feature naming."""
    cfg = DatasetConfig(
        n_informative=2,
        n_noise=1,
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
        n_informative=2,
        n_noise=1,
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
        n_informative=3,
        n_noise=2,
        class_configs=[
            ClassConfig(n_samples=30),
            ClassConfig(n_samples=40),
            ClassConfig(n_samples=30),
        ],
        class_sep=[1.5, 2.0],
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
        n_informative=3,
        n_noise=2,
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
        n_informative=2,
        n_noise=1,
        corr_clusters=[
            CorrClusterConfig(n_cluster_features=4, correlation=0.7),
        ],
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],  # Need 2 classes
    )

    # Pass wrong n_cluster_cols
    with pytest.raises(ValueError, match="Mismatch between x_clusters.shape"):
        _make_names_and_roles(
            cfg,
            n_cluster_cols=5,  # Should be 4
            n_inf_cols=1,  # With anchor, n_inf_free is 1 (2 total - 1 anchor)
            n_noise_cols=1,
        )


def test_make_names_and_roles_mismatch_inf_cols_raises():
    """Test that mismatch in informative columns raises ValueError."""
    cfg = DatasetConfig(
        n_informative=2,
        n_noise=1,
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],  # Need 2 classes
    )

    with pytest.raises(ValueError, match="generate_informative_features must produce"):
        _make_names_and_roles(
            cfg,
            n_cluster_cols=0,
            n_inf_cols=3,  # Should be 2 (n_informative_free)
            n_noise_cols=1,
        )


def test_make_names_and_roles_mismatch_noise_cols_raises():
    """Test that mismatch in noise columns raises ValueError."""
    cfg = DatasetConfig(
        n_informative=2,
        n_noise=1,
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
        n_informative=2,
        n_noise=1,
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],
    )

    # Call with correct parameters - should succeed
    names, inf_idx, noi_idx, cluster_idx, anchor_idx = _make_names_and_roles(
        cfg,
        n_cluster_cols=0,
        n_inf_cols=2,
        n_noise_cols=1,
    )

    assert len(names) == 3


def test_make_names_and_roles_with_informative_anchor():
    """Test naming and roles with informative cluster anchor."""
    cfg = DatasetConfig(
        n_informative=2,
        n_noise=1,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                correlation=0.7,
                anchor_role="informative",
            ),
        ],
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],  # Need 2 classes
        prefixed_feature_naming=True,
    )

    names, inf_idx, noi_idx, cluster_idx, anchor_idx = _make_names_and_roles(
        cfg,
        n_cluster_cols=3,
        n_inf_cols=1,  # With anchor, n_informative_free is 1 (2 total - 1 anchor)
        n_noise_cols=1,
    )

    # Informative should include free informative (1) + anchor (1) = 2
    assert len(inf_idx) == 2
    # Noise should only include free noise (1)
    assert len(noi_idx) == 1
    # Should have one cluster
    assert len(cluster_idx) == 1


def test_make_names_and_roles_with_noise_anchor():
    """Test naming and roles with noise cluster anchor."""
    cfg = DatasetConfig(
        n_informative=2,
        n_noise=1,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                correlation=0.7,
                anchor_role="noise",
            ),
        ],
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],  # Need 2 classes
        prefixed_feature_naming=True,
    )

    names, inf_idx, noi_idx, cluster_idx, anchor_idx = _make_names_and_roles(
        cfg,
        n_cluster_cols=3,
        n_inf_cols=2,  # All 2 informative are free (no anchor)
        n_noise_cols=0,  # With noise anchor in cluster, n_noise_free is 0
    )

    # Informative should only include free informative (2)
    assert len(inf_idx) == 2
    # Noise anchor is not added to noise_idx (noise_idx is only for free noise)
    assert len(noi_idx) == 0  # No free noise when anchor is noise


def test_generate_dataset_metadata_complete():
    """Test that metadata contains all expected fields."""
    cfg = DatasetConfig(
        n_informative=3,
        n_noise=2,
        corr_clusters=[
            CorrClusterConfig(n_cluster_features=4, correlation=0.7),
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
    # Per-cluster correlation structure and strength are exposed first-class.
    assert meta.cluster_structure == {0: "equicorrelated"}
    assert meta.cluster_correlation == {0: 0.7}


def test_metadata_exposes_per_cluster_structure_and_correlation():
    """cluster_structure and cluster_correlation reflect each cluster's config."""
    cfg = DatasetConfig(
        n_informative=2,  # one free informative + one informative anchor
        n_noise=1,  # one noise anchor (no free noise)
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                correlation=0.7,
                structure="equicorrelated",
                anchor_role="informative",
                anchor_effect_size="medium",
                anchor_class=1,
            ),
            CorrClusterConfig(
                n_cluster_features=3,
                correlation=0.6,
                structure="toeplitz",
                anchor_role="noise",
            ),
        ],
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],
        random_state=42,
    )

    _, _, meta = generate_dataset(cfg, return_dataframe=False)

    # One entry per cluster_id, reflecting the configured values.
    assert set(meta.cluster_structure) == {0, 1}
    assert set(meta.cluster_correlation) == {0, 1}
    assert meta.cluster_structure == {0: "equicorrelated", 1: "toeplitz"}
    assert meta.cluster_correlation == {0: 0.7, 1: 0.6}


def test_metadata_preserves_class_specific_correlation():
    """A per-class correlation mapping is preserved unchanged in the metadata."""
    cfg = DatasetConfig(
        n_informative=1,
        n_noise=0,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=4,
                correlation={0: 0.0, 1: 0.8},
                structure="equicorrelated",
                anchor_role="informative",
                anchor_effect_size="medium",
                anchor_class=1,
            ),
        ],
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],
        random_state=42,
    )

    _, _, meta = generate_dataset(cfg, return_dataframe=False)

    assert meta.cluster_structure == {0: "equicorrelated"}
    assert meta.cluster_correlation == {0: {0: 0.0, 1: 0.8}}


def test_generate_dataset_with_no_informative():
    """Test generation with only noise features."""
    cfg = DatasetConfig(
        n_informative=0,
        n_noise=5,
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
        n_informative=5,
        n_noise=0,
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
    """End-to-end generation works when a cluster has a noise anchor.

    The standalone noise block must contain only free noise features; the noise
    anchor and its proxies live inside the correlated-cluster block. The total
    column count therefore equals cfg.n_features rather than including the
    noise anchor twice.
    """
    cfg = DatasetConfig(
        n_informative=2,  # 1 free informative + 1 informative anchor
        n_noise=2,  # 1 noise anchor + 1 free noise
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                correlation=0.7,
                anchor_role="informative",
                anchor_effect_size="medium",
                anchor_class=1,
            ),
            CorrClusterConfig(n_cluster_features=3, correlation=0.5, anchor_role="noise"),
        ],
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],
        random_state=42,
    )

    X, y, meta = generate_dataset(cfg, return_dataframe=False)

    assert X.shape == (100, cfg.n_features) == (100, 8)
    # Noise anchor and proxies stay in the cluster block, not in noise_idx.
    assert meta.noise_idx == [7]
    assert meta.informative_idx == [0, 1]
    assert meta.anchor_role == {0: "informative", 1: "noise"}
    assert meta.anchor_idx == {0: 1, 1: 4}
    assert meta.corr_cluster_indices == {0: [1, 2, 3], 1: [4, 5, 6]}


def test_generate_dataset_populates_batch_meta():
    """generate_dataset stores all batch settings in meta.batch."""
    cfg = DatasetConfig(
        n_informative=5,
        n_noise=3,
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
        n_informative=5,
        n_noise=3,
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
        n_informative=4,
        n_noise=2,
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
        n_informative=3,
        n_noise=2,
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
        n_informative=3,
        n_noise=2,
        class_configs=[ClassConfig(n_samples=25), ClassConfig(n_samples=25)],
        random_state=1,
    )

    _, _, meta = generate_dataset(cfg, return_dataframe=False)

    assert meta.batch is None
    assert meta.batch_labels is None
