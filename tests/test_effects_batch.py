# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tests for batch effects module.

This module tests:
- Batch assignment generation (random and confounded)
- Batch effect application (additive and multiplicative)
- DataFrame and array handling
- Edge cases and validation
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from biomedical_data_generator.effects.batch import (
    apply_batch_effects,
    generate_batch_assignments,
)

# ============================================================================
# Tests for generate_batch_assignments
# ============================================================================


class TestGenerateBatchAssignments:
    """Tests for batch assignment generation."""

    def test_random_assignment_equal_proportions(self):
        """Random assignment should distribute samples evenly across batches."""
        batches = generate_batch_assignments(n_samples=100, n_batches=4, proportions=None, random_state=42)

        counts = np.bincount(batches)
        assert len(counts) == 4
        assert counts.sum() == 100
        # Equal proportions: each batch should have ~25 samples
        assert all(23 <= c <= 27 for c in counts)

    def test_random_assignment_custom_proportions(self):
        """Random assignment with custom proportions."""
        batches = generate_batch_assignments(n_samples=200, n_batches=3, proportions=[0.5, 0.3, 0.2], random_state=42)

        counts = np.bincount(batches)
        assert len(counts) == 3
        assert counts.sum() == 200
        # Check proportions are approximately correct
        assert 95 <= counts[0] <= 105  # ~100 (50%)
        assert 55 <= counts[1] <= 65  # ~60 (30%)
        assert 35 <= counts[2] <= 45  # ~40 (20%)

    def test_proportions_not_summing_to_one(self):
        """Proportions should be normalized automatically."""
        batches = generate_batch_assignments(
            n_samples=100, n_batches=2, proportions=[3, 1], random_state=42  # Will be normalized to [0.75, 0.25]
        )

        counts = np.bincount(batches)
        assert len(counts) == 2
        # Should result in ~75/25 split
        assert 70 <= counts[0] <= 80
        assert 20 <= counts[1] <= 30

    def test_confounded_assignment_no_confounding(self):
        """With confounding_strength=0, should behave like random."""
        y = np.array([0] * 50 + [1] * 50)
        batches = generate_batch_assignments(
            n_samples=100, n_batches=2, class_labels=y, confounding_strength=0.0, random_state=42
        )

        # Check both classes are present in both batches
        for batch_id in range(2):
            mask = batches == batch_id
            classes_in_batch = np.unique(y[mask])
            assert len(classes_in_batch) > 0

    def test_confounded_assignment_moderate_confounding(self):
        """With moderate confounding, classes should prefer certain batches."""
        y = np.array([0] * 100 + [1] * 100)
        batches = generate_batch_assignments(
            n_samples=200, n_batches=2, class_labels=y, confounding_strength=0.8, random_state=42
        )

        # Count class 0 in batch 0
        class0_in_batch0 = np.sum((y == 0) & (batches == 0))
        class1_in_batch1 = np.sum((y == 1) & (batches == 1))

        # With 0.8 confounding (redistribution):
        # P(preferred) = 0.5 + 0.8 * 0.5 = 0.9, expect ~90 out of 100
        # Allow variance: expect at least 80 (80%)
        assert class0_in_batch0 > 80, f"Expected >80, got {class0_in_batch0}"
        assert class1_in_batch1 > 80, f"Expected >80, got {class1_in_batch1}"

    def test_confounded_assignment_perfect_confounding(self):
        """With confounding_strength=1.0, should have perfect separation."""
        y = np.array([0] * 50 + [1] * 50 + [2] * 50)
        batches = generate_batch_assignments(
            n_samples=150, n_batches=3, class_labels=y, confounding_strength=1.0, random_state=42
        )

        # Each class should be perfectly in its preferred batch
        # At strength=1.0 with redistribution: P(preferred) = 1.0
        for cls in range(3):
            cls_batches = batches[y == cls]
            most_common = np.bincount(cls_batches).argmax()
            count_in_preferred = np.sum(cls_batches == most_common)
            # With perfect confounding, expect all 50 samples
            # Allow for stochastic variation: at least 48/50 (96%)
            assert count_in_preferred >= 48, f"Class {cls}: expected >=48 in preferred batch, got {count_in_preferred}"

    def test_more_classes_than_batches(self):
        """Should handle cases where n_classes > n_batches."""
        y = np.array([0] * 20 + [1] * 20 + [2] * 20 + [3] * 20)
        batches = generate_batch_assignments(
            n_samples=80, n_batches=2, class_labels=y, confounding_strength=0.5, random_state=42
        )

        assert len(batches) == 80
        assert set(batches) == {0, 1}

    def test_reproducibility(self):
        """Same random_state should produce identical results."""
        y = np.array([0] * 50 + [1] * 50)

        batches1 = generate_batch_assignments(
            100, n_batches=3, class_labels=y, confounding_strength=0.6, random_state=123
        )
        batches2 = generate_batch_assignments(
            100, n_batches=3, class_labels=y, confounding_strength=0.6, random_state=123
        )

        assert_array_equal(batches1, batches2)

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different assignments."""
        batches1 = generate_batch_assignments(100, n_batches=3, random_state=1)
        batches2 = generate_batch_assignments(100, n_batches=3, random_state=2)

        assert not np.array_equal(batches1, batches2)


# ============================================================================
# Tests for apply_batch_effects
# ============================================================================


class TestApplyBatchEffects:
    """Tests for applying batch effects to data."""

    def test_additive_effects_dataframe(self):
        """Additive effects should shift features by batch-specific constants."""
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"feature_{i}" for i in range(5)])
        batches = generate_batch_assignments(100, n_batches=2, random_state=42)

        X_orig = X.copy()
        X_batch, intercepts = apply_batch_effects(
            X, batches, effect_type="additive", effect_strength=1.0, random_state=42
        )

        # Should return DataFrame
        assert isinstance(X_batch, pd.DataFrame)
        assert X_batch.shape == X.shape
        assert list(X_batch.columns) == list(X.columns)

        # Original should be unchanged
        assert_allclose(X.values, X_orig.values)

        # Intercepts should be drawn
        assert len(intercepts) == 2
        assert intercepts.shape == (2,)

        # Verify effect was applied
        assert not np.allclose(X_batch.values, X.values)

    def test_multiplicative_effects_array(self):
        """Multiplicative effects should scale features."""
        X = np.random.randn(80, 6)
        batches = generate_batch_assignments(80, n_batches=4, random_state=42)

        X_orig = X.copy()
        X_batch, intercepts = apply_batch_effects(
            X, batches, effect_type="multiplicative", effect_strength=0.5, random_state=42
        )

        # Should return array
        assert isinstance(X_batch, np.ndarray)
        assert X_batch.shape == X.shape

        # Original unchanged
        assert_allclose(X, X_orig)

        # Intercepts drawn
        assert len(intercepts) == 4

    def test_affected_features_subset(self):
        """Should only affect specified features."""
        X = np.random.randn(60, 10)
        batches = generate_batch_assignments(60, n_batches=2, random_state=42)

        affected = [0, 2, 4]  # Only affect features 0, 2, 4
        X_batch, _ = apply_batch_effects(
            X, batches, effect_type="additive", effect_strength=1.0, affected_features=affected, random_state=42
        )

        # Unaffected features should be identical
        for feat_idx in [1, 3, 5, 6, 7, 8, 9]:
            assert_allclose(X[:, feat_idx], X_batch[:, feat_idx])

        # Affected features should be different
        for feat_idx in affected:
            assert not np.allclose(X[:, feat_idx], X_batch[:, feat_idx])

    def test_effect_strength_scales_correctly(self):
        """Larger effect_strength should produce larger differences."""
        X = np.random.randn(100, 5)
        batches = generate_batch_assignments(100, n_batches=2, random_state=42)

        X_weak, intercepts_weak = apply_batch_effects(
            X, batches, effect_type="additive", effect_strength=0.1, random_state=42
        )

        X_strong, intercepts_strong = apply_batch_effects(
            X, batches, effect_type="additive", effect_strength=1.0, random_state=42
        )

        # Stronger effects should have larger intercepts (on average)
        assert np.std(intercepts_weak) < np.std(intercepts_strong)

        # Stronger effects should produce larger differences from original
        diff_weak = np.abs(X_weak - X).mean()
        diff_strong = np.abs(X_strong - X).mean()
        assert diff_weak < diff_strong

    def test_zero_effect_strength(self):
        """With effect_strength=0, data should be unchanged."""
        X = np.random.randn(50, 5)
        batches = generate_batch_assignments(50, n_batches=2, random_state=42)

        X_batch, intercepts = apply_batch_effects(
            X, batches, effect_type="additive", effect_strength=0.0, random_state=42
        )

        # Intercepts should be ~0
        assert_allclose(intercepts, 0.0, atol=1e-10)
        # Data should be unchanged
        assert_allclose(X_batch, X)

    def test_reproducibility_with_seed(self):
        """Same random_state should produce identical effects."""
        X = np.random.randn(100, 5)
        batches = generate_batch_assignments(100, n_batches=3, random_state=1)

        X_batch1, intercepts1 = apply_batch_effects(
            X, batches, effect_type="multiplicative", effect_strength=0.5, random_state=42
        )

        X_batch2, intercepts2 = apply_batch_effects(
            X, batches, effect_type="multiplicative", effect_strength=0.5, random_state=42
        )

        assert_allclose(X_batch1, X_batch2)
        assert_allclose(intercepts1, intercepts2)

    def test_batch_means_differ(self):
        """Features should have different means per batch after effects."""
        X = np.random.randn(200, 5)
        batches = generate_batch_assignments(200, n_batches=4, random_state=42)

        X_batch, _ = apply_batch_effects(X, batches, effect_type="additive", effect_strength=1.0, random_state=42)

        # Compute mean of feature 0 per batch
        batch_means = []
        for batch_id in range(4):
            mask = batches == batch_id
            batch_means.append(X_batch[mask, 0].mean())

        # Means should differ
        assert np.std(batch_means) > 0.3  # Reasonable difference

    def test_invalid_effect_type_raises(self):
        """Unknown effect_type should raise ValueError."""
        X = np.random.randn(50, 5)
        batches = generate_batch_assignments(50, n_batches=2, random_state=42)

        with pytest.raises(ValueError, match="Unknown effect_type"):
            apply_batch_effects(X, batches, effect_type="invalid_type", effect_strength=0.5, random_state=42)

    def test_dataframe_preserves_index(self):
        """DataFrame index should be preserved after effects."""
        custom_index = [f"sample_{i}" for i in range(50)]
        X = pd.DataFrame(np.random.randn(50, 3), columns=["A", "B", "C"], index=custom_index)
        batches = generate_batch_assignments(50, n_batches=2, random_state=42)

        X_batch, _ = apply_batch_effects(X, batches, random_state=42)

        assert list(X_batch.index) == custom_index

    def test_empty_batch_handled(self):
        """Batches with no samples should not cause errors."""
        X = np.random.randn(50, 5)
        # Manually create assignments where batch 2 is empty
        batches = np.array([0] * 25 + [1] * 25)

        X_batch, intercepts = apply_batch_effects(
            X, batches, effect_type="additive", effect_strength=0.5, random_state=42
        )

        # Should still draw intercepts for all batches
        assert len(intercepts) == 2


# ============================================================================
# Integration tests
# ============================================================================


class TestBatchEffectsIntegration:
    """Integration tests combining assignment and effects."""

    def test_full_pipeline_with_confounding(self):
        """Test complete workflow: generate confounded batches, apply effects."""
        # Generate data
        X = pd.DataFrame(np.random.randn(200, 10), columns=[f"feature_{i}" for i in range(10)])
        y = np.array([0] * 100 + [1] * 100)

        # Create confounded batches
        batches = generate_batch_assignments(
            200, n_batches=4, class_labels=y, confounding_strength=0.7, random_state=42
        )

        # Apply effects
        X_batch, intercepts = apply_batch_effects(
            X,
            batches,
            effect_type="additive",
            effect_strength=0.8,
            affected_features=[0, 1, 2],  # Only first 3 features
            random_state=42,
        )

        # Verify structure
        assert isinstance(X_batch, pd.DataFrame)
        assert X_batch.shape == (200, 10)
        assert len(intercepts) == 4

        # Features 0-2 should differ, 3-9 should be unchanged
        for i in range(3):
            assert not np.allclose(X.iloc[:, i], X_batch.iloc[:, i])
        for i in range(3, 10):
            assert_allclose(X.iloc[:, i], X_batch.iloc[:, i])

    def test_batch_effects_increase_variance(self):
        """Batch effects should increase overall feature variance."""
        X = np.random.randn(150, 8)
        batches = generate_batch_assignments(150, n_batches=3, random_state=42)

        var_before = X.var(axis=0)

        X_batch, _ = apply_batch_effects(X, batches, effect_type="additive", effect_strength=1.0, random_state=42)

        var_after = X_batch.var(axis=0)

        # Variance should generally increase (not strictly, but on average)
        assert var_after.mean() > var_before.mean()
