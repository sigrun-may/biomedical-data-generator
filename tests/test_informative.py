# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tests for informative features generation."""

import numpy as np
import pytest

from biomedical_data_generator import ClassConfig, DatasetConfig
from biomedical_data_generator.features.informative import (
    _build_class_labels,
    _class_offsets_from_sep,
    _normalize_class_sep,
    generate_informative_features,
    shift_classes,
)


def test_normalize_class_sep_scalar():
    """Test normalization with scalar input."""
    result = _normalize_class_sep(2.0, K=3)
    assert len(result) == 2
    np.testing.assert_array_equal(result, [2.0, 2.0])


def test_normalize_class_sep_sequence():
    """Test normalization with sequence input."""
    result = _normalize_class_sep([1.0, 2.0, 3.0], K=4)
    assert len(result) == 3
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def test_normalize_class_sep_invalid_k_raises():
    """Test that K < 2 raises ValueError."""
    with pytest.raises(ValueError, match="K must be >= 2"):
        _normalize_class_sep(1.0, K=1)


def test_normalize_class_sep_non_finite_scalar_raises():
    """Test that non-finite scalar raises ValueError."""
    with pytest.raises(ValueError, match="must be finite"):
        _normalize_class_sep(np.inf, K=2)


def test_normalize_class_sep_wrong_length_raises():
    """Test that wrong sequence length raises ValueError."""
    with pytest.raises(ValueError, match="length must be K-1"):
        _normalize_class_sep([1.0, 2.0], K=4)  # Need 3 values, got 2


def test_normalize_class_sep_non_1d_raises():
    """Test that multidimensional array raises ValueError."""
    with pytest.raises(ValueError, match="must be 1D"):
        _normalize_class_sep([[1.0, 2.0], [3.0, 4.0]], K=3)


def test_normalize_class_sep_non_finite_entries_raises():
    """Test that non-finite entries raise ValueError."""
    with pytest.raises(ValueError, match="must be finite"):
        _normalize_class_sep([1.0, np.nan, 3.0], K=4)


def test_class_offsets_from_sep():
    """Test computing class offsets from separations."""
    sep_vec = np.array([2.0, 3.0])
    offsets = _class_offsets_from_sep(sep_vec)

    # Should have K=3 offsets (K-1=2 separations)
    assert len(offsets) == 3

    # Should be mean-centered
    assert np.abs(offsets.mean()) < 1e-10

    # Check cumulative nature
    assert offsets[1] - offsets[0] == 2.0
    assert offsets[2] - offsets[1] == 3.0


def test_build_class_labels():
    """Test building class labels from config."""
    cfg = DatasetConfig(
        n_informative=1,
        n_noise=0,
        class_configs=[
            ClassConfig(n_samples=5),
            ClassConfig(n_samples=3),
            ClassConfig(n_samples=7),
        ],
        class_sep=[1.0, 1.5],  # Need n_classes - 1 = 2 values
    )

    y = _build_class_labels(cfg)

    assert len(y) == 15  # 5 + 3 + 7
    assert np.sum(y == 0) == 5
    assert np.sum(y == 1) == 3
    assert np.sum(y == 2) == 7


def test_shift_classes_empty_array():
    """Test shift_classes with empty array."""
    X = np.empty((0, 0))
    y = np.array([])

    # Should not raise error
    shift_classes(X, y, informative_idx=[], class_sep=[1.0])


def test_shift_classes_empty_y():
    """Test shift_classes with non-empty X but empty y."""
    X = np.ones((0, 3))
    y = np.array([])

    # Should not raise error
    shift_classes(X, y, informative_idx=[0, 1, 2], class_sep=[1.0])


def test_shift_classes_single_class():
    """Test shift_classes with K=1 (no shifting needed)."""
    X = np.ones((10, 3))
    y = np.zeros(10, dtype=int)

    X_before = X.copy()

    shift_classes(X, y, informative_idx=[0, 1, 2], class_sep=[1.0])

    # Should not change X when there's only one class
    np.testing.assert_array_equal(X, X_before)


def test_shift_classes_invalid_y_shape_raises():
    """Test that multidimensional y raises ValueError."""
    X = np.ones((10, 3))
    y = np.zeros((10, 1))  # Should be 1D

    with pytest.raises(ValueError, match="must be a 1D label array"):
        shift_classes(X, y, informative_idx=[0, 1, 2], class_sep=[1.0])


def test_shift_classes_with_anchors_equalized():
    """Test shift_classes with anchor contributions in equalized mode."""
    X = np.zeros((100, 5))
    y = np.array([0] * 50 + [1] * 50)

    anchor_contrib = {
        0: (1.0, 0),  # Anchor at column 0, target class 0
    }

    shift_classes(
        X,
        y,
        informative_idx=[1, 2, 3, 4],
        anchor_contrib=anchor_contrib,
        class_sep=[2.0],
        anchor_strength=1.0,
        anchor_mode="equalized",
    )

    # Anchor column 0 should have different values for each class
    assert X[y == 0, 0].mean() != X[y == 1, 0].mean()


def test_shift_classes_with_anchors_strong():
    """Test shift_classes with anchor contributions in strong mode."""
    X = np.zeros((100, 5))
    y = np.array([0] * 50 + [1] * 50)

    anchor_contrib = {
        0: (1.0, 0),  # Anchor at column 0, target class 0
    }

    shift_classes(
        X,
        y,
        informative_idx=[1, 2, 3, 4],
        anchor_contrib=anchor_contrib,
        class_sep=[2.0],
        anchor_strength=1.0,
        anchor_mode="strong",
    )

    # Anchor column 0 should have different values for each class
    assert X[y == 0, 0].mean() > X[y == 1, 0].mean()


def test_shift_classes_invalid_anchor_mode_raises():
    """Test that invalid anchor_mode raises ValueError."""
    X = np.zeros((10, 3))
    y = np.array([0] * 5 + [1] * 5)

    anchor_contrib = {0: (1.0, 0)}

    with pytest.raises(ValueError, match="Unknown anchor_mode"):
        shift_classes(
            X,
            y,
            informative_idx=[1, 2],
            anchor_contrib=anchor_contrib,
            class_sep=[2.0],
            anchor_mode="invalid_mode",
        )


def test_shift_classes_anchor_class_out_of_range_raises():
    """Test that anchor target class out of range raises ValueError."""
    X = np.zeros((10, 3))
    y = np.array([0] * 5 + [1] * 5)

    anchor_contrib = {0: (1.0, 5)}  # Class 5 doesn't exist

    with pytest.raises(ValueError, match="out of range"):
        shift_classes(
            X,
            y,
            informative_idx=[1, 2],
            anchor_contrib=anchor_contrib,
            class_sep=[2.0],
        )


def test_shift_classes_spread_non_anchors_false():
    """Test shift_classes with spread_non_anchors=False."""
    X = np.zeros((100, 5))
    y = np.array([0] * 50 + [1] * 50)

    X_before = X.copy()

    shift_classes(
        X,
        y,
        informative_idx=[0, 1, 2, 3, 4],
        anchor_contrib=None,
        class_sep=[2.0],
        spread_non_anchors=False,
    )

    # X should remain unchanged when spread_non_anchors=False and no anchors
    np.testing.assert_array_equal(X, X_before)


def test_shift_classes_multiclass():
    """Test shift_classes with multiple classes."""
    X = np.zeros((150, 3))
    y = np.array([0] * 50 + [1] * 50 + [2] * 50)

    shift_classes(
        X,
        y,
        informative_idx=[0, 1, 2],
        anchor_contrib=None,
        class_sep=[1.0, 2.0],
    )

    # Each class should have different means
    mean_0 = X[y == 0].mean(axis=0)
    mean_1 = X[y == 1].mean(axis=0)
    mean_2 = X[y == 2].mean(axis=0)

    assert not np.allclose(mean_0, mean_1)
    assert not np.allclose(mean_1, mean_2)


def test_generate_informative_features_basic():
    """Test basic informative feature generation."""
    cfg = DatasetConfig(
        n_informative=3,
        n_noise=0,
        class_configs=[
            ClassConfig(n_samples=50),
            ClassConfig(n_samples=50),
        ],
        random_state=42,
    )

    rng = np.random.default_rng(42)
    X, y = generate_informative_features(cfg, rng)

    assert X.shape == (100, 3)
    assert len(y) == 100
    assert np.sum(y == 0) == 50
    assert np.sum(y == 1) == 50


def test_generate_informative_features_no_informative():
    """Test generation with n_informative_free=0."""
    cfg = DatasetConfig(
        n_informative=0,
        n_noise=5,
        class_configs=[
            ClassConfig(n_samples=20),
            ClassConfig(n_samples=30),
        ],
        random_state=42,
    )

    rng = np.random.default_rng(42)
    X, y = generate_informative_features(cfg, rng)

    # Should return empty matrix for features
    assert X.shape == (50, 0)
    assert len(y) == 50


def test_generate_informative_features_multiclass():
    """Test generation with multiple classes."""
    cfg = DatasetConfig(
        n_informative=5,
        n_noise=0,
        class_configs=[
            ClassConfig(n_samples=30),
            ClassConfig(n_samples=40),
            ClassConfig(n_samples=30),
        ],
        class_sep=[1.5, 2.0],
        random_state=42,
    )

    rng = np.random.default_rng(42)
    X, y = generate_informative_features(cfg, rng)

    assert X.shape == (100, 5)
    assert len(y) == 100
    assert np.sum(y == 0) == 30
    assert np.sum(y == 1) == 40
    assert np.sum(y == 2) == 30

    # Classes should be separated
    mean_0 = X[y == 0].mean(axis=0)
    mean_1 = X[y == 1].mean(axis=0)
    mean_2 = X[y == 2].mean(axis=0)

    assert not np.allclose(mean_0, mean_1)
    assert not np.allclose(mean_1, mean_2)


def test_generate_informative_features_different_distributions():
    """Test generation with different per-class distributions."""
    cfg = DatasetConfig(
        n_informative=3,
        n_noise=0,
        class_configs=[
            ClassConfig(
                n_samples=50,
                class_distribution="normal",
                class_distribution_params={"loc": 0.0, "scale": 1.0},
            ),
            ClassConfig(
                n_samples=50,
                class_distribution="uniform",
                class_distribution_params={"low": 5.0, "high": 10.0},
            ),
        ],
        class_sep=2.0,
        random_state=42,
    )

    rng = np.random.default_rng(42)
    X, y = generate_informative_features(cfg, rng)

    assert X.shape == (100, 3)

    # Class 1 samples (before shifting) should be in [5, 10] range roughly
    # Hard to test exactly due to shifting, but we can check basic properties


def test_shift_classes_anchor_column_mean_preserved():
    """Test that anchor columns preserve global mean."""
    X = np.ones((100, 3)) * 5.0  # Start with mean=5.0
    y = np.array([0] * 50 + [1] * 50)

    anchor_contrib = {0: (1.0, 0)}

    shift_classes(
        X,
        y,
        informative_idx=[1, 2],
        anchor_contrib=anchor_contrib,
        class_sep=[2.0],
    )

    # Global mean of anchor column should be approximately preserved
    assert np.abs(X[:, 0].mean() - 5.0) < 0.1
