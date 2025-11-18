# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Critical MVP tests for production readiness."""

import numpy as np
import pytest

from biomedical_data_generator import CorrClusterConfig, DatasetConfig, generate_dataset


def test_feature_names_match_dataframe_columns():
    """Verify meta.feature_names exactly matches DataFrame columns."""
    cfg = DatasetConfig(
        n_samples=50,
        n_features=5,
        n_informative=2,
        n_noise=3,
        n_classes=2,
        class_counts={0: 25, 1: 25},
        feature_naming="prefixed",
        random_state=42,
    )
    X_df, y, meta = generate_dataset(cfg, return_dataframe=True)

    # Check exact match
    assert list(X_df.columns) == meta.feature_names
    assert len(meta.feature_names) == cfg.n_features

    # Check naming scheme
    assert meta.feature_names[0].startswith("i")  # informative
    assert any(name.startswith("n") for name in meta.feature_names)  # noise


def test_anchor_improves_classification_accuracy():
    """Verify that informative anchors actually improve classification.

    Compare dataset with informative anchor vs. without.
    The informative anchor should lead to better separability.
    """
    # Dataset WITH informative anchor (large effect)
    cfg_with = DatasetConfig(
        n_samples=200,
        n_informative=1,
        n_noise=3,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                rho=0.7,
                structure="equicorrelated",
                anchor_role="informative",
                anchor_effect_size="large",  # 1.5
                anchor_class=1,
            )
        ],
        n_features=1 + 3 + (3 - 1),
        n_classes=2,
        class_counts={0: 100, 1: 100},
        class_sep=1.0,
        random_state=42,
    )

    # Dataset WITHOUT informative anchor (all pseudo) - use "noise" role instead
    cfg_without = DatasetConfig(
        n_samples=200,
        n_informative=1,
        n_noise=3,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                rho=0.7,
                structure="equicorrelated",
                anchor_role="noise",  # NOT informative
            )
        ],
        n_features=1 + 3 + 3,  # noise anchor: all 3 features are proxies
        n_classes=2,
        class_counts={0: 100, 1: 100},
        class_sep=1.0,
        random_state=42,
    )

    X_with, y_with, meta_with = generate_dataset(cfg_with, return_dataframe=False)
    X_without, y_without, meta_without = generate_dataset(cfg_without, return_dataframe=False)

    # Simple separability test: compute class centroids distance
    # With informative anchor, centroids should be further apart
    def centroid_distance(X, y):
        c0 = X[y == 0].mean(axis=0)
        c1 = X[y == 1].mean(axis=0)
        return float(np.linalg.norm(c0 - c1))

    dist_with = centroid_distance(X_with, y_with)
    dist_without = centroid_distance(X_without, y_without)

    # Informative anchor should create larger separation
    assert dist_with > dist_without, (
        f"Informative anchor should increase class separation. "
        f"Got: with={dist_with:.2f}, without={dist_without:.2f}"
    )


def test_class_specific_correlation_in_clusters():
    """Verify class_rho creates class-specific correlations.

    Features should be more correlated within target class samples.
    Labels are now generated BEFORE features, so class_rho works correctly.
    """
    cfg = DatasetConfig(
        n_samples=200,
        n_informative=1,
        n_noise=0,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=4,
                structure="equicorrelated",
                anchor_role="informative",
                class_rho={1: 0.9},  # Class 1: high correlation
                rho_baseline=0.1,  # Other classes: low correlation
            )
        ],
        n_features=4,
        n_classes=2,
        class_counts={0: 100, 1: 100},
        random_state=123,
    )

    X, y, meta = generate_dataset(cfg, return_dataframe=False)

    # Extract cluster features
    cluster_cols = meta.corr_cluster_indices[1]
    X_cluster = X[:, cluster_cols]

    # Compute correlation within each class
    X_class0 = X_cluster[y == 0]
    X_class1 = X_cluster[y == 1]

    corr0 = np.corrcoef(X_class0, rowvar=False)
    corr1 = np.corrcoef(X_class1, rowvar=False)

    # Extract off-diagonal correlations (upper triangle)
    mask = np.triu(np.ones_like(corr0, dtype=bool), k=1)
    mean_corr0 = float(corr0[mask].mean())
    mean_corr1 = float(corr1[mask].mean())

    # Class 1 should have higher correlation due to class_rho={1: 0.9}
    # Allow some tolerance due to finite sample size
    assert mean_corr1 > mean_corr0 + 0.3, (
        f"Class 1 should have higher correlation. " f"Got: class0={mean_corr0:.2f}, class1={mean_corr1:.2f}"
    )

    # Class 1 should be close to 0.9
    assert 0.7 < mean_corr1 < 1.0, f"Class 1 correlation should be ~0.9, got {mean_corr1:.2f}"


def test_metadata_completeness():
    """Verify DatasetMeta contains all required fields."""
    cfg = DatasetConfig(
        n_samples=100,
        n_features=8,
        n_informative=3,
        n_noise=5,
        n_classes=2,
        class_counts={0: 50, 1: 50},
        random_state=42,
    )
    X, y, meta = generate_dataset(cfg)

    # Required fields
    assert hasattr(meta, "feature_names")
    assert hasattr(meta, "informative_idx")
    assert hasattr(meta, "noise_idx")
    assert hasattr(meta, "y_counts")
    assert hasattr(meta, "y_weights")
    assert hasattr(meta, "n_classes")
    assert hasattr(meta, "class_sep")
    assert hasattr(meta, "random_state")
    assert hasattr(meta, "resolved_config")

    # Check types
    assert isinstance(meta.feature_names, list)
    assert isinstance(meta.informative_idx, list)
    assert isinstance(meta.y_counts, dict)
    assert isinstance(meta.y_weights, tuple)

    # Check values
    assert len(meta.feature_names) == cfg.n_features
    assert meta.n_classes == cfg.n_classes
    assert meta.y_counts == {0: 50, 1: 50}
    assert sum(meta.y_weights) == pytest.approx(1.0)


def test_single_sample_per_class():
    """Edge case: Minimum viable dataset (1 sample per class)."""
    cfg = DatasetConfig(
        n_samples=3,
        n_features=4,
        n_informative=2,
        n_noise=2,
        n_classes=3,
        class_counts={0: 1, 1: 1, 2: 1},
        random_state=42,
    )
    X, y, meta = generate_dataset(cfg, return_dataframe=False)

    assert X.shape == (3, 4)
    assert len(y) == 3
    assert set(y) == {0, 1, 2}
    assert meta.y_counts == {0: 1, 1: 1, 2: 1}


def test_many_classes():
    """Multi-class with 10+ classes."""
    n_classes = 12
    samples_per_class = 10
    cfg = DatasetConfig(
        n_samples=n_classes * samples_per_class,
        n_features=6,
        n_informative=3,
        n_noise=3,
        n_classes=n_classes,
        class_counts={i: samples_per_class for i in range(n_classes)},
        random_state=42,
    )
    X, y, meta = generate_dataset(cfg, return_dataframe=False)

    assert X.shape == (120, 6)
    assert len(set(y)) == n_classes
    assert meta.n_classes == n_classes
    assert all(meta.y_counts[i] == samples_per_class for i in range(n_classes))


def test_extreme_noise_scales():
    """Test very small and very large noise scales."""
    # Very small noise
    cfg_small = DatasetConfig(
        n_samples=100,
        n_features=5,
        n_informative=2,
        n_noise=3,
        n_classes=2,
        class_counts={0: 50, 1: 50},
        noise_scale=0.01,
        random_state=42,
    )
    X_small, y_small, meta_small = generate_dataset(cfg_small, return_dataframe=False)
    noise_cols = meta_small.noise_idx
    noise_std_small = float(X_small[:, noise_cols].std())
    assert noise_std_small < 0.1, f"Small noise should have std < 0.1, got {noise_std_small:.3f}"

    # Very large noise
    cfg_large = DatasetConfig(
        n_samples=100,
        n_features=5,
        n_informative=2,
        n_noise=3,
        n_classes=2,
        class_counts={0: 50, 1: 50},
        noise_scale=100.0,
        random_state=42,
    )
    X_large, y_large, meta_large = generate_dataset(cfg_large, return_dataframe=False)
    noise_cols = meta_large.noise_idx
    noise_std_large = float(X_large[:, noise_cols].std())
    assert noise_std_large > 50, f"Large noise should have std > 50, got {noise_std_large:.1f}"


def test_imbalanced_dataset_90_10():
    """Realistic medical scenario: 90% healthy, 10% diseased."""
    cfg = DatasetConfig(
        n_samples=1000,
        n_features=8,
        n_informative=3,
        n_noise=5,
        n_classes=2,
        class_counts={0: 900, 1: 100},  # 90/10 split
        random_state=42,
    )
    X, y, meta = generate_dataset(cfg, return_dataframe=False)

    assert meta.y_counts == {0: 900, 1: 100}
    assert len(y[y == 0]) == 900
    assert len(y[y == 1]) == 100

    # Check imbalance ratio
    ratio = meta.y_counts[0] / meta.y_counts[1]
    assert ratio == pytest.approx(9.0)
