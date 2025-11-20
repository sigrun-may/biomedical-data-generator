# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Comprehensive tests for informative.py module."""

import numpy as np
import pytest

from biomedical_data_generator.config import ClassConfig, DatasetConfig
from biomedical_data_generator.features.informative import (
    _build_class_labels,
    _class_offsets_from_sep,
    _normalize_class_sep,
    generate_informative_features,
)


class TestNormalizeClassSep:
    """Test _normalize_class_sep function."""

    def test_scalar_broadcast(self):
        """Test that scalar class_sep is broadcast to (K-1) length."""
        result = _normalize_class_sep(2.5, K=3)
        expected = np.array([2.5, 2.5])
        np.testing.assert_array_equal(result, expected)

    def test_scalar_broadcast_k2(self):
        """Test scalar broadcast for K=2."""
        result = _normalize_class_sep(1.0, K=2)
        expected = np.array([1.0])
        np.testing.assert_array_equal(result, expected)

    def test_sequence_input(self):
        """Test that sequence input is converted to array."""
        result = _normalize_class_sep([1.0, 2.0, 3.0], K=4)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_sequence_correct_length(self):
        """Test that sequence must have length K-1."""
        # Should work
        result = _normalize_class_sep([1.0, 2.0], K=3)
        assert result.shape == (2,)

    def test_k_less_than_2_raises(self):
        """Test that K < 2 raises ValueError."""
        with pytest.raises(ValueError, match="K must be >= 2"):
            _normalize_class_sep(1.0, K=1)

        with pytest.raises(ValueError, match="K must be >= 2"):
            _normalize_class_sep(1.0, K=0)

    def test_non_finite_scalar_raises(self):
        """Test that non-finite scalar raises ValueError."""
        with pytest.raises(ValueError, match="class_sep must be finite"):
            _normalize_class_sep(np.inf, K=3)

        with pytest.raises(ValueError, match="class_sep must be finite"):
            _normalize_class_sep(np.nan, K=3)

    def test_wrong_sequence_length_raises(self):
        """Test that sequence with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="class_sep length must be K-1"):
            _normalize_class_sep([1.0, 2.0], K=4)  # needs length 3

        with pytest.raises(ValueError, match="class_sep length must be K-1"):
            _normalize_class_sep([1.0], K=3)  # needs length 2

    def test_non_finite_sequence_raises(self):
        """Test that non-finite sequence entries raise ValueError."""
        with pytest.raises(ValueError, match="class_sep entries must be finite"):
            _normalize_class_sep([1.0, np.inf], K=3)

        with pytest.raises(ValueError, match="class_sep entries must be finite"):
            _normalize_class_sep([np.nan, 1.0], K=3)

    def test_multidimensional_sequence_raises(self):
        """Test that multidimensional input raises ValueError."""
        with pytest.raises(ValueError, match="class_sep must be 1D"):
            _normalize_class_sep([[1.0, 2.0], [3.0, 4.0]], K=3)

    def test_zero_separation(self):
        """Test that zero separation is allowed."""
        result = _normalize_class_sep(0.0, K=3)
        expected = np.array([0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_negative_separation(self):
        """Test that negative separations are allowed."""
        result = _normalize_class_sep([-1.0, 2.0], K=3)
        expected = np.array([-1.0, 2.0])
        np.testing.assert_array_equal(result, expected)


class TestClassOffsetsFromSep:
    """Test _class_offsets_from_sep function."""

    def test_uniform_separations(self):
        """Test cumulative offsets with uniform separations."""
        sep_vec = np.array([1.0, 1.0, 1.0])  # K=4
        result = _class_offsets_from_sep(sep_vec)

        # Cumulative: [0, 1, 2, 3], mean = 1.5, centered: [-1.5, -0.5, 0.5, 1.5]
        expected = np.array([-1.5, -0.5, 0.5, 1.5])
        np.testing.assert_allclose(result, expected)

    def test_non_uniform_separations(self):
        """Test cumulative offsets with non-uniform separations."""
        sep_vec = np.array([1.0, 2.0])  # K=3
        result = _class_offsets_from_sep(sep_vec)

        # Cumulative: [0, 1, 3], mean = 4/3, centered: [-4/3, -1/3, 5/3]
        expected = np.array([-4 / 3, -1 / 3, 5 / 3])
        np.testing.assert_allclose(result, expected)

    def test_single_separation(self):
        """Test with K=2 (single separation)."""
        sep_vec = np.array([2.0])
        result = _class_offsets_from_sep(sep_vec)

        # Cumulative: [0, 2], mean = 1, centered: [-1, 1]
        expected = np.array([-1.0, 1.0])
        np.testing.assert_allclose(result, expected)

    def test_zero_separations(self):
        """Test with zero separations (all classes at same location)."""
        sep_vec = np.array([0.0, 0.0])  # K=3
        result = _class_offsets_from_sep(sep_vec)

        # Cumulative: [0, 0, 0], mean = 0, centered: [0, 0, 0]
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected)

    def test_mean_zero_property(self):
        """Test that result always has mean zero."""
        test_cases = [
            np.array([1.0]),
            np.array([1.0, 2.0]),
            np.array([0.5, 1.5, 2.5]),
            np.array([10.0, 20.0, 30.0, 40.0]),
        ]

        for sep_vec in test_cases:
            result = _class_offsets_from_sep(sep_vec)
            np.testing.assert_allclose(result.mean(), 0.0, atol=1e-10)

    def test_correct_length(self):
        """Test that output has length K when input has length K-1."""
        for K in range(2, 10):
            sep_vec = np.ones(K - 1)
            result = _class_offsets_from_sep(sep_vec)
            assert result.shape == (K,)

    def test_negative_separations(self):
        """Test with negative separations."""
        sep_vec = np.array([-1.0, 2.0])  # K=3
        result = _class_offsets_from_sep(sep_vec)

        # Cumulative: [0, -1, 1], mean = 0, centered: [0, -1, 1]
        expected = np.array([0.0, -1.0, 1.0])
        np.testing.assert_allclose(result, expected)


class TestBuildClassLabels:
    """Test _build_class_labels function."""

    def test_balanced_two_classes(self):
        """Test label construction for balanced two-class dataset."""
        cfg = DatasetConfig(
            n_informative=3,
            n_noise=2,
            class_configs=[
                ClassConfig(n_samples=10, class_distribution="normal"),
                ClassConfig(n_samples=10, class_distribution="normal"),
            ],
        )

        y = _build_class_labels(cfg)
        assert y.shape == (20,)
        assert np.array_equal(y[:10], np.zeros(10))
        assert np.array_equal(y[10:], np.ones(10))

    def test_imbalanced_classes(self):
        """Test label construction for imbalanced dataset."""
        cfg = DatasetConfig(
            n_informative=3,
            n_noise=2,
            class_sep=[1.0, 1.0],  # K=3, need length K-1=2
            class_configs=[
                ClassConfig(n_samples=5, class_distribution="normal"),
                ClassConfig(n_samples=15, class_distribution="normal"),
                ClassConfig(n_samples=10, class_distribution="normal"),
            ],
        )

        y = _build_class_labels(cfg)
        assert y.shape == (30,)
        assert np.sum(y == 0) == 5
        assert np.sum(y == 1) == 15
        assert np.sum(y == 2) == 10

    def test_single_sample_per_class(self):
        """Test with single sample per class."""
        cfg = DatasetConfig(
            n_informative=3,
            n_noise=2,
            class_sep=[1.0, 1.0, 1.0],  # K=4, need length K-1=3
            class_configs=[
                ClassConfig(n_samples=1, class_distribution="normal"),
                ClassConfig(n_samples=1, class_distribution="normal"),
                ClassConfig(n_samples=1, class_distribution="normal"),
                ClassConfig(n_samples=1, class_distribution="normal"),
            ],
        )

        y = _build_class_labels(cfg)
        expected = np.array([0, 1, 2, 3])
        np.testing.assert_array_equal(y, expected)

    def test_label_dtype_is_integer(self):
        """Test that labels are integer type."""
        cfg = DatasetConfig(
            n_informative=3,
            n_noise=2,
            class_configs=[
                ClassConfig(n_samples=5, class_distribution="normal"),
                ClassConfig(n_samples=5, class_distribution="normal"),
            ],
        )

        y = _build_class_labels(cfg)
        assert y.dtype.kind in ("i", "u")

    def test_correct_label_range(self):
        """Test that labels are in range [0, K-1]."""
        K = 5
        cfg = DatasetConfig(
            n_informative=3,
            n_noise=2,
            class_sep=[1.0] * (K - 1),  # K=5, need length K-1=4
            class_configs=[ClassConfig(n_samples=10, class_distribution="normal") for _ in range(K)],
        )

        y = _build_class_labels(cfg)
        assert y.min() == 0
        assert y.max() == K - 1
        assert len(np.unique(y)) == K


class TestGenerateInformativeFeatures:
    """Test generate_informative_features function."""

    def test_basic_generation(self):
        """Test basic informative feature generation."""
        cfg = DatasetConfig(
            n_informative=3,
            n_noise=2,
            random_state=42,
            class_configs=[
                ClassConfig(n_samples=10, class_distribution="normal"),
                ClassConfig(n_samples=10, class_distribution="normal"),
            ],
        )

        rng = np.random.default_rng(42)
        X, y = generate_informative_features(cfg, rng)

        assert X.shape == (20, 3)
        assert y.shape == (20,)
        assert np.sum(y == 0) == 10
        assert np.sum(y == 1) == 10

    def test_no_informative_features(self):
        """Test with zero informative features."""
        cfg = DatasetConfig(
            n_informative=0,
            n_noise=5,
            random_state=42,
            class_configs=[
                ClassConfig(n_samples=10, class_distribution="normal"),
                ClassConfig(n_samples=10, class_distribution="normal"),
            ],
        )

        rng = np.random.default_rng(42)
        X, y = generate_informative_features(cfg, rng)

        assert X.shape == (20, 0)
        assert y.shape == (20,)

    def test_class_separation_applied(self):
        """Test that class separation is applied correctly."""
        cfg = DatasetConfig(
            n_informative=2,
            n_noise=3,
            class_sep=3.0,
            random_state=42,
            class_configs=[
                ClassConfig(n_samples=20, class_distribution="normal", class_distribution_params={"loc": 0, "scale": 0.1}),
                ClassConfig(n_samples=20, class_distribution="normal", class_distribution_params={"loc": 0, "scale": 0.1}),
            ],
        )

        rng = np.random.default_rng(42)
        X, y = generate_informative_features(cfg, rng)

        # Classes should be separated
        mean_cls0 = X[y == 0].mean(axis=0)
        mean_cls1 = X[y == 1].mean(axis=0)

        # With small within-class variance, separation should be close to class_sep
        separation = np.abs(mean_cls0 - mean_cls1)
        assert np.all(separation > 2.0)  # Should be well separated

    def test_multiclass_generation(self):
        """Test generation with multiple classes."""
        K = 4
        cfg = DatasetConfig(
            n_informative=3,
            n_noise=2,
            class_sep=[1.0] * (K - 1),  # K=4, need length K-1=3
            random_state=42,
            class_configs=[ClassConfig(n_samples=10, class_distribution="normal") for _ in range(K)],
        )

        rng = np.random.default_rng(42)
        X, y = generate_informative_features(cfg, rng)

        assert X.shape == (40, 3)
        assert y.shape == (40,)
        assert len(np.unique(y)) == K

        for k in range(K):
            assert np.sum(y == k) == 10

    def test_reproducibility(self):
        """Test that generation is reproducible with same seed."""
        cfg = DatasetConfig(
            n_informative=4,
            n_noise=1,
            class_sep=[1.0, 1.0],  # K=3, need length K-1=2
            random_state=123,
            class_configs=[ClassConfig(n_samples=10, class_distribution="normal") for _ in range(3)],
        )

        rng1 = np.random.default_rng(123)
        X1, y1 = generate_informative_features(cfg, rng1)

        rng2 = np.random.default_rng(123)
        X2, y2 = generate_informative_features(cfg, rng2)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_distributions_per_class(self):
        """Test with different distributions for each class."""
        cfg = DatasetConfig(
            n_informative=2,
            n_noise=3,
            class_sep=[1.0, 1.0],  # K=3, need length K-1=2
            random_state=42,
            class_configs=[
                ClassConfig(n_samples=10, class_distribution="normal"),
                ClassConfig(n_samples=10, class_distribution="uniform", class_distribution_params={"low": -1, "high": 1}),
                ClassConfig(n_samples=10, class_distribution="laplace"),
            ],
        )

        rng = np.random.default_rng(42)
        X, y = generate_informative_features(cfg, rng)

        assert X.shape == (30, 2)
        assert y.shape == (30,)
        # Just verify it runs without error - distribution differences are tested elsewhere

    def test_mean_centering(self):
        """Test that overall mean of each feature is approximately zero after shifting."""
        cfg = DatasetConfig(
            n_informative=3,
            n_noise=2,
            class_sep=[2.0, 2.0],  # K=3, need length K-1=2
            random_state=42,
            class_configs=[ClassConfig(n_samples=100, class_distribution="normal") for _ in range(3)],
        )

        rng = np.random.default_rng(42)
        X, y = generate_informative_features(cfg, rng)

        # Overall mean should be close to zero for each feature
        overall_means = X.mean(axis=0)
        np.testing.assert_allclose(overall_means, 0.0, atol=0.5)

    def test_imbalanced_classes(self):
        """Test generation with imbalanced class sizes."""
        cfg = DatasetConfig(
            n_informative=2,
            n_noise=3,
            class_sep=[1.0, 1.0],  # K=3, need length K-1=2
            random_state=42,
            class_configs=[
                ClassConfig(n_samples=5, class_distribution="normal"),
                ClassConfig(n_samples=20, class_distribution="normal"),
                ClassConfig(n_samples=10, class_distribution="normal"),
            ],
        )

        rng = np.random.default_rng(42)
        X, y = generate_informative_features(cfg, rng)

        assert X.shape == (35, 2)
        assert np.sum(y == 0) == 5
        assert np.sum(y == 1) == 20
        assert np.sum(y == 2) == 10

    def test_two_classes_minimal(self):
        """Test with minimal two classes (K=2)."""
        cfg = DatasetConfig(
            n_informative=3,
            n_noise=2,
            random_state=42,
            class_configs=[
                ClassConfig(n_samples=10, class_distribution="normal"),
                ClassConfig(n_samples=10, class_distribution="normal"),
            ],
        )

        rng = np.random.default_rng(42)
        X, y = generate_informative_features(cfg, rng)

        assert X.shape == (20, 3)
        assert y.shape == (20,)
        assert np.sum(y == 0) == 10
        assert np.sum(y == 1) == 10

    def test_custom_class_distribution_params(self):
        """Test with custom distribution parameters per class."""
        cfg = DatasetConfig(
            n_informative=2,
            n_noise=3,
            random_state=42,
            class_configs=[
                ClassConfig(
                    n_samples=10,
                    class_distribution="normal",
                    class_distribution_params={"loc": 0.0, "scale": 0.5},
                ),
                ClassConfig(
                    n_samples=10,
                    class_distribution="normal",
                    class_distribution_params={"loc": 0.0, "scale": 2.0},
                ),
            ],
        )

        rng = np.random.default_rng(42)
        X, y = generate_informative_features(cfg, rng)

        # Class 1 should have higher variance before shifting
        # (After shifting, means will differ, but base variance pattern should be visible)
        assert X.shape == (20, 2)
        assert y.shape == (20,)
