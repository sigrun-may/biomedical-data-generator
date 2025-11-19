# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Comprehensive tests for shift_classes() function."""

import numpy as np
import pytest

from biomedical_data_generator.features.informative import shift_classes


class TestShiftClassesEdgeCases:
    """Test edge cases for shift_classes()."""

    def test_empty_array(self):
        """Test with empty input arrays."""
        X = np.array([]).reshape(0, 5)
        y = np.array([])
        shift_classes(X, y, informative_idx=[0, 1])
        # Should not raise any errors
        assert X.shape == (0, 5)

    def test_single_class(self):
        """Test with single class (K=1) - no shifting should occur."""
        X = np.ones((10, 3))
        X_orig = X.copy()
        y = np.zeros(10, dtype=int)

        shift_classes(
            X, y,
            informative_idx=[0, 1],
            anchor_contrib={2: (1.0, 0)},
            class_sep=2.0
        )

        # With K=1, no shifting should occur
        np.testing.assert_array_equal(X, X_orig)

    def test_no_informative_features(self):
        """Test with empty informative_idx."""
        X = np.ones((20, 3))
        y = np.array([0] * 10 + [1] * 10)

        shift_classes(
            X, y,
            informative_idx=[],
            class_sep=1.0
        )

        # Only check that no error is raised
        assert X.shape == (20, 3)

    def test_no_anchor_contrib(self):
        """Test without anchor contributions."""
        X = np.zeros((20, 3))
        y = np.array([0] * 10 + [1] * 10)

        shift_classes(
            X, y,
            informative_idx=[0, 1],
            anchor_contrib=None,
            class_sep=1.0,
            spread_non_anchors=True
        )

        # Should spread non-anchors without errors
        # Class 0 should have negative shifts, class 1 positive
        assert np.mean(X[y == 0, 0]) < 0
        assert np.mean(X[y == 1, 0]) > 0

    def test_invalid_anchor_mode(self):
        """Test with invalid anchor_mode parameter."""
        X = np.zeros((20, 3))
        y = np.array([0] * 10 + [1] * 10)

        with pytest.raises(ValueError, match="Unknown anchor_mode"):
            shift_classes(
                X, y,
                informative_idx=[0],
                anchor_contrib={1: (1.0, 0)},
                anchor_mode="invalid_mode"
            )

    def test_spread_non_anchors_disabled(self):
        """Test with spread_non_anchors=False."""
        X = np.zeros((30, 4))
        y = np.array([0] * 10 + [1] * 10 + [2] * 10)

        shift_classes(
            X, y,
            informative_idx=[0, 1, 2],
            anchor_contrib=None,
            class_sep=1.0,
            spread_non_anchors=False
        )

        # Without spreading, X should remain zeros
        np.testing.assert_array_equal(X, np.zeros((30, 4)))

    def test_zero_class_sep(self):
        """Test with class_sep=0 (no separation)."""
        X = np.zeros((20, 3))
        y = np.array([0] * 10 + [1] * 10)

        shift_classes(
            X, y,
            informative_idx=[0, 1],
            anchor_contrib={2: (1.0, 0)},
            class_sep=0.0
        )

        # With class_sep=0, no shifts should occur
        np.testing.assert_array_equal(X, np.zeros((20, 3)))

    def test_zero_anchor_strength(self):
        """Test with anchor_strength=0."""
        X = np.zeros((20, 3))
        y = np.array([0] * 10 + [1] * 10)

        shift_classes(
            X, y,
            informative_idx=[0],
            anchor_contrib={1: (1.0, 0)},
            anchor_strength=0.0,
            spread_non_anchors=False
        )

        # With anchor_strength=0, anchors should have no effect
        np.testing.assert_array_equal(X, np.zeros((20, 3)))


class TestShiftClassesAnchorModes:
    """Test different anchor modes: 'equalized' and 'strong'."""

    def test_equalized_mode_two_classes(self):
        """Test 'equalized' anchor mode with K=2."""
        X = np.zeros((20, 3))
        y = np.array([0] * 10 + [1] * 10)

        shift_classes(
            X, y,
            informative_idx=[0],
            anchor_contrib={1: (1.0, 0)},
            class_sep=2.0,
            anchor_strength=1.0,
            anchor_mode="equalized",
            spread_non_anchors=False
        )

        # For equalized mode with K=2: A = class_sep * anchor_strength * beta * (K-1)/K
        # A = 2.0 * 1.0 * 1.0 * 1/2 = 1.0
        # Class 0 gets +A, class 1 gets -A/(K-1) = -A/1 = -1.0
        expected_cls0 = 1.0
        expected_cls1 = -1.0

        np.testing.assert_allclose(X[y == 0, 1], expected_cls0, rtol=1e-10)
        np.testing.assert_allclose(X[y == 1, 1], expected_cls1, rtol=1e-10)

    def test_equalized_mode_three_classes(self):
        """Test 'equalized' anchor mode with K=3."""
        X = np.zeros((30, 3))
        y = np.array([0] * 10 + [1] * 10 + [2] * 10)

        shift_classes(
            X, y,
            informative_idx=[0],
            anchor_contrib={1: (1.0, 0)},
            class_sep=2.0,
            anchor_strength=1.0,
            anchor_mode="equalized",
            spread_non_anchors=False
        )

        # For equalized mode with K=3: A = 2.0 * 1.0 * 1.0 * 2/3 = 4/3
        # Class 0 gets +A, classes 1 and 2 get -A/(K-1) = -A/2 = -2/3
        A = 2.0 * 1.0 * 1.0 * 2 / 3
        expected_cls0 = A
        expected_other = -A / 2

        np.testing.assert_allclose(X[y == 0, 1], expected_cls0, rtol=1e-10)
        np.testing.assert_allclose(X[y == 1, 1], expected_other, rtol=1e-10)
        np.testing.assert_allclose(X[y == 2, 1], expected_other, rtol=1e-10)

    def test_strong_mode_two_classes(self):
        """Test 'strong' anchor mode with K=2."""
        X = np.zeros((20, 3))
        y = np.array([0] * 10 + [1] * 10)

        shift_classes(
            X, y,
            informative_idx=[0],
            anchor_contrib={1: (1.0, 0)},
            class_sep=2.0,
            anchor_strength=1.0,
            anchor_mode="strong",
            spread_non_anchors=False
        )

        # For strong mode with K=2: A = class_sep * anchor_strength * beta * (K-1)/2
        # A = 2.0 * 1.0 * 1.0 * 1/2 = 1.0
        A = 2.0 * 1.0 * 1.0 * 1 / 2
        expected_cls0 = A
        expected_cls1 = -A / 1

        np.testing.assert_allclose(X[y == 0, 1], expected_cls0, rtol=1e-10)
        np.testing.assert_allclose(X[y == 1, 1], expected_cls1, rtol=1e-10)

    def test_strong_mode_three_classes(self):
        """Test 'strong' anchor mode with K=3."""
        X = np.zeros((30, 3))
        y = np.array([0] * 10 + [1] * 10 + [2] * 10)

        shift_classes(
            X, y,
            informative_idx=[0],
            anchor_contrib={1: (1.0, 0)},
            class_sep=2.0,
            anchor_strength=1.0,
            anchor_mode="strong",
            spread_non_anchors=False
        )

        # For strong mode with K=3: A = 2.0 * 1.0 * 1.0 * 2/2 = 2.0
        A = 2.0 * 1.0 * 1.0 * 2 / 2
        expected_cls0 = A
        expected_other = -A / 2

        np.testing.assert_allclose(X[y == 0, 1], expected_cls0, rtol=1e-10)
        np.testing.assert_allclose(X[y == 1, 1], expected_other, rtol=1e-10)
        np.testing.assert_allclose(X[y == 2, 1], expected_other, rtol=1e-10)

    def test_equalized_vs_strong_scaling_with_k(self):
        """Test that equalized mode scales differently than strong mode as K increases."""
        K_values = [2, 3, 4, 5]
        equalized_strengths = []
        strong_strengths = []

        for K in K_values:
            # Test equalized mode
            X_eq = np.zeros((K * 10, 2))
            y = np.repeat(np.arange(K), 10)
            shift_classes(
                X_eq, y,
                informative_idx=[],
                anchor_contrib={0: (1.0, 0)},
                class_sep=1.0,
                anchor_strength=1.0,
                anchor_mode="equalized",
                spread_non_anchors=False
            )
            equalized_strengths.append(np.mean(X_eq[y == 0, 0]))

            # Test strong mode
            X_strong = np.zeros((K * 10, 2))
            shift_classes(
                X_strong, y,
                informative_idx=[],
                anchor_contrib={0: (1.0, 0)},
                class_sep=1.0,
                anchor_strength=1.0,
                anchor_mode="strong",
                spread_non_anchors=False
            )
            strong_strengths.append(np.mean(X_strong[y == 0, 0]))

        # For equalized: A = (K-1)/K, which approaches 1 as K increases
        # For strong: A = (K-1)/2, which grows linearly with K

        # Strong mode should grow more than equalized mode
        assert strong_strengths[-1] > equalized_strengths[-1] * 2


class TestShiftClassesBetaAndStrength:
    """Test different beta values and anchor strengths."""

    def test_different_beta_values(self):
        """Test anchors with different beta values."""
        X = np.zeros((20, 4))
        y = np.array([0] * 10 + [1] * 10)

        shift_classes(
            X, y,
            informative_idx=[],
            anchor_contrib={
                0: (0.5, 0),  # beta=0.5
                1: (1.0, 0),  # beta=1.0
                2: (2.0, 1),  # beta=2.0
            },
            class_sep=1.0,
            anchor_strength=1.0,
            anchor_mode="equalized",
            spread_non_anchors=False
        )

        # Higher beta should result in stronger shifts
        shift_col0 = np.abs(np.mean(X[y == 0, 0]))
        shift_col1 = np.abs(np.mean(X[y == 0, 1]))
        shift_col2 = np.abs(np.mean(X[y == 1, 2]))

        assert shift_col0 < shift_col1  # beta 0.5 < beta 1.0
        assert shift_col1 < shift_col2  # beta 1.0 < beta 2.0

    def test_different_anchor_strengths(self):
        """Test different anchor_strength values."""
        strengths = [0.5, 1.0, 2.0]
        mean_shifts = []

        for strength in strengths:
            X = np.zeros((20, 2))
            y = np.array([0] * 10 + [1] * 10)

            shift_classes(
                X, y,
                informative_idx=[],
                anchor_contrib={0: (1.0, 0)},
                class_sep=1.0,
                anchor_strength=strength,
                anchor_mode="equalized",
                spread_non_anchors=False
            )

            mean_shifts.append(np.abs(np.mean(X[y == 0, 0])))

        # Higher anchor_strength should result in larger shifts
        assert mean_shifts[0] < mean_shifts[1] < mean_shifts[2]


class TestShiftClassesNonAnchorSpread:
    """Test non-anchor spreading behavior."""

    def test_two_class_spread(self):
        """Test spreading of non-anchor features across two classes."""
        X = np.zeros((20, 3))
        y = np.array([0] * 10 + [1] * 10)

        shift_classes(
            X, y,
            informative_idx=[0, 1],
            anchor_contrib=None,
            class_sep=2.0,
            spread_non_anchors=True
        )

        # For K=2: spread_vec = class_sep * ([0, 1] - 0.5) / 1 = 2.0 * [-0.5, 0.5] = [-1.0, 1.0]
        expected_cls0 = -1.0
        expected_cls1 = 1.0

        np.testing.assert_allclose(X[y == 0, 0], expected_cls0, rtol=1e-10)
        np.testing.assert_allclose(X[y == 1, 0], expected_cls1, rtol=1e-10)
        np.testing.assert_allclose(X[y == 0, 1], expected_cls0, rtol=1e-10)
        np.testing.assert_allclose(X[y == 1, 1], expected_cls1, rtol=1e-10)

    def test_three_class_spread(self):
        """Test spreading of non-anchor features across three classes."""
        X = np.zeros((30, 2))
        y = np.array([0] * 10 + [1] * 10 + [2] * 10)

        shift_classes(
            X, y,
            informative_idx=[0, 1],
            anchor_contrib=None,
            class_sep=2.0,
            spread_non_anchors=True
        )

        # For K=3: spread_vec = 2.0 * ([0, 1, 2] - 1.0) / 2 = 2.0 * [-1, 0, 1] / 2 = [-1, 0, 1]
        expected_cls0 = -1.0
        expected_cls1 = 0.0
        expected_cls2 = 1.0

        np.testing.assert_allclose(X[y == 0, 0], expected_cls0, rtol=1e-10)
        np.testing.assert_allclose(X[y == 1, 0], expected_cls1, rtol=1e-10)
        np.testing.assert_allclose(X[y == 2, 0], expected_cls2, rtol=1e-10)

    def test_anchors_excluded_from_spread(self):
        """Test that anchor features are not affected by non-anchor spreading."""
        X = np.zeros((20, 3))
        y = np.array([0] * 10 + [1] * 10)

        shift_classes(
            X, y,
            informative_idx=[0, 1, 2],  # All informative, but col 1 is anchor
            anchor_contrib={1: (1.0, 0)},
            class_sep=1.0,
            anchor_strength=1.0,
            anchor_mode="equalized",
            spread_non_anchors=True
        )

        # Columns 0 and 2 should have spread applied
        # Column 1 should only have anchor shift (not spread)

        # For K=2, spread is [-0.5, 0.5]
        spread_cls0 = -0.5
        spread_cls1 = 0.5

        # Check non-anchor columns have spread
        np.testing.assert_allclose(X[y == 0, 0], spread_cls0, rtol=1e-10)
        np.testing.assert_allclose(X[y == 1, 0], spread_cls1, rtol=1e-10)

        # Check anchor column has only anchor shift (not spread)
        # For equalized with K=2: A = 1.0 * 1.0 * 1.0 * 1/2 = 0.5
        A = 0.5
        np.testing.assert_allclose(X[y == 0, 1], A, rtol=1e-10)
        np.testing.assert_allclose(X[y == 1, 1], -A, rtol=1e-10)


class TestShiftClassesCombinedEffects:
    """Test combined effects of anchors and non-anchor spreading."""

    def test_anchors_and_spread_combined(self):
        """Test that anchor and spread effects combine correctly."""
        X = np.zeros((30, 4))
        y = np.array([0] * 10 + [1] * 10 + [2] * 10)

        shift_classes(
            X, y,
            informative_idx=[0, 1, 2, 3],
            anchor_contrib={1: (1.0, 0), 3: (0.5, 2)},
            class_sep=1.0,
            anchor_strength=1.0,
            anchor_mode="equalized",
            spread_non_anchors=True
        )

        # Column 0, 2 should have spread (non-anchors)
        # For K=3: spread = [-0.5, 0, 0.5]
        np.testing.assert_allclose(X[y == 0, 0], -0.5, rtol=1e-10)
        np.testing.assert_allclose(X[y == 1, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(X[y == 2, 0], 0.5, rtol=1e-10)

        # Column 1 should have anchor effect for class 0
        # A = 1.0 * 1.0 * 1.0 * 2/3
        A1 = 1.0 * 1.0 * 1.0 * 2 / 3
        np.testing.assert_allclose(X[y == 0, 1], A1, rtol=1e-10)
        np.testing.assert_allclose(X[y == 1, 1], -A1 / 2, rtol=1e-10)

        # Column 3 should have anchor effect for class 2
        # A = 1.0 * 1.0 * 0.5 * 2/3
        A3 = 1.0 * 1.0 * 0.5 * 2 / 3
        np.testing.assert_allclose(X[y == 2, 3], A3, rtol=1e-10)
        np.testing.assert_allclose(X[y == 0, 3], -A3 / 2, rtol=1e-10)

    def test_in_place_modification(self):
        """Test that X is modified in place."""
        X = np.zeros((20, 3))
        X_id = id(X)
        y = np.array([0] * 10 + [1] * 10)

        shift_classes(
            X, y,
            informative_idx=[0, 1],
            class_sep=1.0
        )

        # Should modify in place (same object)
        assert id(X) == X_id
        # Should have made changes
        assert not np.allclose(X, 0.0)


class TestShiftClassesMultipleClasses:
    """Test with various numbers of classes."""

    def test_four_classes_equalized(self):
        """Test equalized mode with K=4."""
        X = np.zeros((40, 2))
        y = np.repeat(np.arange(4), 10)

        shift_classes(
            X, y,
            informative_idx=[],
            anchor_contrib={0: (1.0, 1)},
            class_sep=2.0,
            anchor_strength=1.0,
            anchor_mode="equalized",
            spread_non_anchors=False
        )

        # For K=4: A = 2.0 * 1.0 * 1.0 * 3/4 = 1.5
        A = 1.5
        np.testing.assert_allclose(X[y == 1, 0], A, rtol=1e-10)
        np.testing.assert_allclose(X[y == 0, 0], -A / 3, rtol=1e-10)
        np.testing.assert_allclose(X[y == 2, 0], -A / 3, rtol=1e-10)
        np.testing.assert_allclose(X[y == 3, 0], -A / 3, rtol=1e-10)

    def test_five_classes_spread(self):
        """Test non-anchor spread with K=5."""
        X = np.zeros((50, 2))
        y = np.repeat(np.arange(5), 10)

        shift_classes(
            X, y,
            informative_idx=[0, 1],
            anchor_contrib=None,
            class_sep=1.0,
            spread_non_anchors=True
        )

        # For K=5: spread_vec = ([0,1,2,3,4] - 2) / 4 = [-0.5, -0.25, 0, 0.25, 0.5]
        expected = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])

        for k in range(5):
            np.testing.assert_allclose(X[y == k, 0], expected[k], rtol=1e-10)


class TestShiftClassesSumToZero:
    """Test that shifts preserve zero-mean property (sum to zero across classes)."""

    def test_anchor_shifts_sum_to_zero(self):
        """Test that anchor shifts sum to zero across all samples."""
        X = np.zeros((30, 2))
        y = np.array([0] * 10 + [1] * 10 + [2] * 10)

        shift_classes(
            X, y,
            informative_idx=[],
            anchor_contrib={0: (1.0, 1)},
            class_sep=1.0,
            anchor_strength=1.0,
            anchor_mode="equalized",
            spread_non_anchors=False
        )

        # Sum across all samples should be close to zero (within numerical precision)
        np.testing.assert_allclose(np.sum(X[:, 0]), 0.0, atol=1e-10)

    def test_spread_shifts_sum_to_zero(self):
        """Test that spread shifts sum to zero across all samples."""
        X = np.zeros((40, 2))
        y = np.repeat(np.arange(4), 10)

        shift_classes(
            X, y,
            informative_idx=[0, 1],
            anchor_contrib=None,
            class_sep=2.0,
            spread_non_anchors=True
        )

        # Sum across all samples should be close to zero
        np.testing.assert_allclose(np.sum(X[:, 0]), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.sum(X[:, 1]), 0.0, atol=1e-10)

    def test_combined_shifts_sum_to_zero(self):
        """Test that combined anchor and spread shifts sum to zero."""
        X = np.zeros((30, 3))
        y = np.array([0] * 10 + [1] * 10 + [2] * 10)

        shift_classes(
            X, y,
            informative_idx=[0, 1, 2],
            anchor_contrib={1: (1.5, 0)},
            class_sep=1.5,
            anchor_strength=2.0,
            anchor_mode="strong",
            spread_non_anchors=True
        )

        # All columns should sum to approximately zero
        for col in range(3):
            np.testing.assert_allclose(np.sum(X[:, col]), 0.0, atol=1e-10)
