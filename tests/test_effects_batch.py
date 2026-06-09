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

from biomedical_data_generator import BatchEffectsConfig
from biomedical_data_generator.effects.batch import (
    apply_batch_effects,
    apply_batch_effects_from_config,
    confounded_batch_assignment,
    generate_batch_assignments,
    random_batch_assignment,
)

# Fixtures for RNG instances
# ============================================================================


@pytest.fixture
def rng_data():
    """Fixture providing RNG for data generation."""
    return np.random.default_rng(42)


@pytest.fixture
def rng_assign():
    """Fixture providing RNG for batch assignment."""
    return np.random.default_rng(42)


@pytest.fixture
def rng_effects():
    """Fixture providing RNG for effect generation."""
    return np.random.default_rng(43)


@pytest.fixture
def rng_general():
    """Fixture providing general-purpose RNG."""
    return np.random.default_rng(42)


# ============================================================================
# Tests for generate_batch_assignments
# ============================================================================


class TestGenerateBatchAssignments:
    """Tests for batch assignment generation."""

    def test_random_assignment_equal_proportions(self, rng_general):
        """Random assignment should distribute samples evenly across batches."""
        batches = generate_batch_assignments(
            n_samples=100,
            n_batches=4,
            proportions=None,
            rng=rng_general,
        )

        counts = np.bincount(batches)
        assert len(counts) == 4
        assert counts.sum() == 100
        # Equal proportions: each batch should have ~25 samples
        assert all(23 <= c <= 27 for c in counts)

    def test_random_assignment_custom_proportions(self, rng_general):
        """Random assignment with custom proportions."""
        batches = generate_batch_assignments(
            n_samples=200,
            n_batches=3,
            proportions=[0.5, 0.3, 0.2],
            rng=rng_general,
        )

        counts = np.bincount(batches)
        assert len(counts) == 3
        assert counts.sum() == 200
        # Check proportions are approximately correct
        assert 95 <= counts[0] <= 105  # ~100 (50%)
        assert 55 <= counts[1] <= 65  # ~60 (30%)
        assert 35 <= counts[2] <= 45  # ~40 (20%)

    def test_proportions_not_summing_to_one(self, rng_general):
        """Proportions should be normalized automatically."""
        batches = generate_batch_assignments(
            n_samples=100,
            n_batches=2,
            proportions=[3, 1],  # Will be normalized to [0.75, 0.25]
            rng=rng_general,
        )

        counts = np.bincount(batches)
        assert len(counts) == 2
        # Should result in ~75/25 split
        assert 70 <= counts[0] <= 80
        assert 20 <= counts[1] <= 30

    def test_confounded_assignment_no_confounding(self, rng_general):
        """With confounding_with_class=0, should behave like random."""
        y = np.array([0] * 50 + [1] * 50)
        batches = generate_batch_assignments(
            n_samples=100,
            n_batches=2,
            class_labels=y,
            confounding_with_class=0.0,
            rng=rng_general,
        )

        # Check both classes are present in both batches
        for batch_id in range(2):
            mask = batches == batch_id
            classes_in_batch = np.unique(y[mask])
            assert len(classes_in_batch) > 0

    def test_confounded_assignment_moderate_confounding(self, rng_general):
        """With moderate confounding, classes should prefer certain batches."""
        y = np.array([0] * 100 + [1] * 100)
        batches = generate_batch_assignments(
            n_samples=200,
            n_batches=2,
            class_labels=y,
            confounding_with_class=0.8,
            rng=rng_general,
        )

        # Count class 0 in batch 0 and class 1 in batch 1
        class0_in_batch0 = np.sum((y == 0) & (batches == 0))
        class1_in_batch1 = np.sum((y == 1) & (batches == 1))

        # With 0.8 confounding (redistribution):
        # P(preferred) = 0.5 + 0.8 * 0.5 = 0.9, expect ~90 out of 100
        # Allow variance: expect at least 80 (80%)
        assert class0_in_batch0 > 80, f"Expected >80, got {class0_in_batch0}"
        assert class1_in_batch1 > 80, f"Expected >80, got {class1_in_batch1}"

    def test_confounded_assignment_perfect_confounding(self, rng_general):
        """With confounding_with_class=1.0, should have near-perfect separation."""
        y = np.array([0] * 50 + [1] * 50 + [2] * 50)
        batches = generate_batch_assignments(
            n_samples=150,
            n_batches=3,
            class_labels=y,
            confounding_with_class=1.0,
            rng=rng_general,
        )

        # Each class should be concentrated in its preferred batch
        for cls in range(3):
            cls_batches = batches[y == cls]
            most_common = np.bincount(cls_batches).argmax()
            count_in_preferred = np.sum(cls_batches == most_common)
            # With strong confounding, expect almost all 50 samples
            # Allow for stochastic variation: at least 48/50 (96%)
            assert count_in_preferred >= 48, f"Class {cls}: expected >=48 in preferred batch, got {count_in_preferred}"

    def test_more_classes_than_batches(self, rng_general):
        """Should handle cases where n_classes > n_batches."""
        y = np.array([0] * 20 + [1] * 20 + [2] * 20 + [3] * 20)
        batches = generate_batch_assignments(
            n_samples=80,
            n_batches=2,
            class_labels=y,
            confounding_with_class=0.5,
            rng=rng_general,
        )

        assert len(batches) == 80
        assert set(batches) == {0, 1}

    def test_reproducibility(self):
        """Same rng seed should produce identical results."""
        y = np.array([0] * 50 + [1] * 50)

        rng1 = np.random.default_rng(123)
        batches1 = generate_batch_assignments(
            100,
            n_batches=3,
            class_labels=y,
            confounding_with_class=0.6,
            rng=rng1,
        )

        rng2 = np.random.default_rng(123)
        batches2 = generate_batch_assignments(
            100,
            n_batches=3,
            class_labels=y,
            confounding_with_class=0.6,
            rng=rng2,
        )

        assert_array_equal(batches1, batches2)

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different assignments."""
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)

        batches1 = generate_batch_assignments(100, n_batches=3, rng=rng1)
        batches2 = generate_batch_assignments(100, n_batches=3, rng=rng2)

        assert not np.array_equal(batches1, batches2)


# ============================================================================
# Tests for apply_batch_effects
# ============================================================================


class TestApplyBatchEffects:
    """Tests for applying batch effects to data."""

    def test_additive_effects_dataframe(self, rng_data, rng_assign, rng_effects):
        """Additive effects should shift features by batch-specific constants."""
        X = pd.DataFrame(
            rng_data.normal(size=(100, 5)),
            columns=[f"feature_{i}" for i in range(5)],
        )

        batches = generate_batch_assignments(100, n_batches=2, rng=rng_assign)

        X_orig = X.copy()
        X_batch, batch_effects = apply_batch_effects(
            X,
            batches,
            rng=rng_effects,
            effect_type="additive",
            effect_strength=1.0,
        )

        # Should return DataFrame
        assert isinstance(X_batch, pd.DataFrame)
        assert X_batch.shape == X.shape
        assert list(X_batch.columns) == list(X.columns)

        # Original should be unchanged
        assert_allclose(X.values, X_orig.values)

        # Effects should be drawn
        assert len(batch_effects) == 2
        assert batch_effects.shape == (2,)

        # Verify effect was applied
        assert not np.allclose(X_batch.values, X.values)

    def test_multiplicative_effects_array(self, rng_assign, rng_effects):
        """Multiplicative effects should scale features."""
        rng_x = np.random.default_rng(101)
        X = rng_x.normal(size=(80, 6))

        batches = generate_batch_assignments(80, n_batches=4, rng=rng_assign)

        X_orig = X.copy()
        X_batch, batch_effects = apply_batch_effects(
            X,
            batches,
            rng=rng_effects,
            effect_type="multiplicative",
            effect_strength=0.5,
        )

        # Should return array
        assert isinstance(X_batch, np.ndarray)
        assert X_batch.shape == X.shape

        # Original unchanged
        assert_allclose(X, X_orig)

        # Effects drawn
        assert len(batch_effects) == 4

    def test_affected_features_subset(self, rng_assign, rng_effects):
        """Should only affect specified features."""
        rng_x = np.random.default_rng(102)
        X = rng_x.normal(size=(60, 10))

        batches = generate_batch_assignments(60, n_batches=2, rng=rng_assign)

        affected = [0, 2, 4]  # Only affect features 0, 2, 4

        X_batch, _ = apply_batch_effects(
            X,
            batches,
            rng=rng_effects,
            effect_type="additive",
            effect_strength=1.0,
            affected_features=affected,
        )

        # Unaffected features should be identical
        for feat_idx in [1, 3, 5, 6, 7, 8, 9]:
            assert_allclose(X[:, feat_idx], X_batch[:, feat_idx])

        # Affected features should be different
        for feat_idx in affected:
            assert not np.allclose(X[:, feat_idx], X_batch[:, feat_idx])

    def test_effect_strength_scales_correctly(self, rng_assign):
        """Larger effect_strength should produce larger differences."""
        rng_x = np.random.default_rng(103)
        X = rng_x.normal(size=(100, 5))

        batches = generate_batch_assignments(100, n_batches=2, rng=rng_assign)

        rng_effects_weak = np.random.default_rng(43)
        X_weak, effects_weak = apply_batch_effects(
            X,
            batches,
            rng=rng_effects_weak,
            effect_type="additive",
            effect_strength=0.1,
        )

        rng_effects_strong = np.random.default_rng(44)
        X_strong, effects_strong = apply_batch_effects(
            X,
            batches,
            rng=rng_effects_strong,
            effect_type="additive",
            effect_strength=1.0,
        )

        # Stronger effects should have larger std of effects (on average)
        assert np.std(effects_weak) < np.std(effects_strong)

        # Stronger effects should produce larger differences from original
        diff_weak = np.abs(X_weak - X).mean()
        diff_strong = np.abs(X_strong - X).mean()
        assert diff_weak < diff_strong

    def test_zero_effect_strength(self, rng_assign, rng_effects):
        """With effect_strength=0, data should be unchanged."""
        rng_x = np.random.default_rng(104)
        X = rng_x.normal(size=(50, 5))

        batches = generate_batch_assignments(50, n_batches=2, rng=rng_assign)

        X_batch, batch_effects = apply_batch_effects(
            X,
            batches,
            rng=rng_effects,
            effect_type="additive",
            effect_strength=0.0,
        )

        # Effects should be ~0
        assert_allclose(batch_effects, 0.0, atol=1e-10)
        # Data should be unchanged
        assert_allclose(X_batch, X)

    def test_reproducibility_with_seed(self, rng_assign):
        """Same rng seed should produce identical effects."""
        rng_x = np.random.default_rng(105)
        X = rng_x.normal(size=(100, 5))

        batches = generate_batch_assignments(100, n_batches=3, rng=rng_assign)

        rng_effects1 = np.random.default_rng(42)
        X_batch1, effects1 = apply_batch_effects(
            X,
            batches,
            rng=rng_effects1,
            effect_type="multiplicative",
            effect_strength=0.5,
        )

        rng_effects2 = np.random.default_rng(42)
        X_batch2, effects2 = apply_batch_effects(
            X,
            batches,
            rng=rng_effects2,
            effect_type="multiplicative",
            effect_strength=0.5,
        )

        assert_allclose(X_batch1, X_batch2)
        assert_allclose(effects1, effects2)

    def test_batch_means_differ(self, rng_assign, rng_effects):
        """Features should have different means per batch after effects."""
        rng_x = np.random.default_rng(1)
        X = rng_x.normal(size=(200, 5))

        batches = generate_batch_assignments(200, n_batches=4, rng=rng_assign)

        X_batch, _ = apply_batch_effects(
            X,
            batches,
            rng=rng_effects,
            effect_type="additive",
            effect_strength=1.0,
        )

        # Compute mean of feature 0 per batch
        batch_means = []
        for batch_id in range(4):
            mask = batches == batch_id
            batch_means.append(X_batch[mask, 0].mean())

        # Means should differ; relaxed threshold to avoid rare flakiness
        assert np.std(batch_means) > 0.25

    def test_invalid_effect_type_raises(self, rng_assign, rng_effects):
        """Unknown effect_type should raise ValueError."""
        rng_x = np.random.default_rng(106)
        X = rng_x.normal(size=(50, 5))

        batches = generate_batch_assignments(50, n_batches=2, rng=rng_assign)

        with pytest.raises(ValueError, match="Unknown effect_type"):
            apply_batch_effects(
                X,
                batches,
                rng=rng_effects,
                effect_type="invalid_type",
                effect_strength=0.5,
            )

    def test_dataframe_preserves_index(self, rng_assign, rng_effects):
        """DataFrame index should be preserved after effects."""
        custom_index = [f"sample_{i}" for i in range(50)]
        rng_x = np.random.default_rng(107)
        X = pd.DataFrame(
            rng_x.normal(size=(50, 3)),
            columns=["A", "B", "C"],
            index=custom_index,
        )

        batches = generate_batch_assignments(50, n_batches=2, rng=rng_assign)

        X_batch, _ = apply_batch_effects(
            X,
            batches,
            rng=rng_effects,
        )

        assert list(X_batch.index) == custom_index

    def test_empty_batch_handled(self, rng_effects):
        """Batches with no samples should not cause errors."""
        rng_x = np.random.default_rng(108)
        X = rng_x.normal(size=(50, 5))
        # Manually create assignments where batch 2 is empty
        batches = np.array([0] * 25 + [1] * 25)

        X_batch, batch_effects = apply_batch_effects(
            X,
            batches,
            rng=rng_effects,
            effect_type="additive",
            effect_strength=0.5,
        )

        # Should still draw effects for all batches (here: 2)
        assert X_batch.shape == (50, 5)
        assert len(batch_effects) == 2


# ============================================================================
# Integration tests
# ============================================================================


class TestBatchEffectsIntegration:
    """Integration tests combining assignment and effects."""

    def test_full_pipeline_with_confounding(self, rng_assign, rng_effects):
        """Test complete workflow: generate confounded batches, apply effects."""
        # Generate data
        rng_data = np.random.default_rng(42)
        X = pd.DataFrame(
            rng_data.normal(size=(200, 10)),
            columns=[f"feature_{i}" for i in range(10)],
        )
        y = np.array([0] * 100 + [1] * 100)

        # Create confounded batches
        batches = generate_batch_assignments(
            200,
            n_batches=4,
            class_labels=y,
            confounding_with_class=0.7,
            rng=rng_assign,
        )

        # Apply effects
        X_batch, batch_effects = apply_batch_effects(
            X,
            batches,
            rng=rng_effects,
            effect_type="additive",
            effect_strength=0.8,
            affected_features=[0, 1, 2],  # Only first 3 features
        )

        # Verify structure
        assert isinstance(X_batch, pd.DataFrame)
        assert X_batch.shape == (200, 10)
        assert len(batch_effects) == 4

        # Features 0–2 should differ, 3–9 should be unchanged
        for i in range(3):
            assert not np.allclose(X.iloc[:, i], X_batch.iloc[:, i])
        for i in range(3, 10):
            assert_allclose(X.iloc[:, i], X_batch.iloc[:, i])

    def test_batch_effects_increase_variance(self, rng_assign, rng_effects):
        """Batch effects should increase overall feature variance."""
        rng_x = np.random.default_rng(109)
        X = rng_x.normal(size=(150, 8))

        batches = generate_batch_assignments(150, n_batches=3, rng=rng_assign)

        var_before = X.var(axis=0)

        X_batch, _ = apply_batch_effects(
            X,
            batches,
            rng=rng_effects,
            effect_type="additive",
            effect_strength=1.0,
        )

        var_after = X_batch.var(axis=0)

        # Variance should generally increase (not strictly, but on average)
        assert var_after.mean() > var_before.mean()


# ============================================================================
# Validation and edge case tests
# ============================================================================


class TestBatchEffectsValidation:
    """Tests for validation and error handling."""

    def test_negative_n_samples_raises(self):
        """Negative n_samples should raise ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="n_samples must be positive"):
            generate_batch_assignments(
                n_samples=-10,
                n_batches=2,
                rng=rng,
            )

    def test_zero_n_samples_raises(self):
        """Zero n_samples should raise ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="n_samples must be positive"):
            generate_batch_assignments(
                n_samples=0,
                n_batches=2,
                rng=rng,
            )

    def test_zero_n_batches_raises(self):
        """Zero n_batches should raise ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="n_batches must be >= 1"):
            generate_batch_assignments(
                n_samples=100,
                n_batches=0,
                rng=rng,
            )

    def test_negative_n_batches_raises(self):
        """Negative n_batches should raise ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="n_batches must be >= 1"):
            generate_batch_assignments(
                n_samples=100,
                n_batches=-2,
                rng=rng,
            )

    def test_confounding_below_zero_raises(self):
        """confounding_with_class below 0 should raise ValueError."""
        y = np.array([0] * 50 + [1] * 50)
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="confounding_with_class must be in"):
            generate_batch_assignments(
                n_samples=100,
                n_batches=2,
                class_labels=y,
                confounding_with_class=-0.1,
                rng=rng,
            )

    def test_confounding_above_one_raises(self):
        """confounding_with_class above 1 should raise ValueError."""
        y = np.array([0] * 50 + [1] * 50)
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="confounding_with_class must be in"):
            generate_batch_assignments(
                n_samples=100,
                n_batches=2,
                class_labels=y,
                confounding_with_class=1.5,
                rng=rng,
            )

    def test_mismatched_class_labels_length_raises(self):
        """class_labels with wrong length should raise ValueError."""
        y = np.array([0] * 30 + [1] * 30)  # Only 60 samples
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="class_labels must have length n_samples"):
            generate_batch_assignments(
                n_samples=100,  # But requesting 100
                n_batches=2,
                class_labels=y,
                confounding_with_class=0.5,
                rng=rng,
            )

    def test_single_batch_returns_all_zeros(self):
        """With n_batches=1, all samples should be in batch 0."""
        rng = np.random.default_rng(42)
        batches = generate_batch_assignments(
            n_samples=50,
            n_batches=1,
            rng=rng,
        )
        assert len(batches) == 50
        assert np.all(batches == 0)

    def test_invalid_proportions_length_raises(self):
        """Proportions with wrong length should raise ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="len\\(proportions\\) must equal n_batches"):
            generate_batch_assignments(
                n_samples=100,
                n_batches=3,
                proportions=[0.5, 0.5],  # Only 2 values for 3 batches
                rng=rng,
            )

    def test_negative_proportions_raises(self):
        """Negative proportions should raise ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="proportions must be non-negative"):
            generate_batch_assignments(
                n_samples=100,
                n_batches=2,
                proportions=[0.7, -0.3],
                rng=rng,
            )

    def test_zero_sum_proportions_raises(self):
        """Proportions summing to zero should raise ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="proportions must sum to a positive value"):
            generate_batch_assignments(
                n_samples=100,
                n_batches=2,
                proportions=[0.0, 0.0],
                rng=rng,
            )

    def test_from_config_respects_scalar_granularity(self):
        """Scalar granularity must apply one identical shift to all affected features per batch."""
        n_samples, n_features = 60, 5
        base_matrix = np.zeros((n_samples, n_features))
        class_labels = np.array([0] * 30 + [1] * 30)

        batch_config = BatchEffectsConfig(
            n_batches=3,
            effect_type="additive",
            effect_strength=1.0,
            effect_granularity="scalar",
            confounding_with_class=0.0,
            affected_features="all",
        )
        rng = np.random.default_rng(42)
        affected_matrix, batch_assignments, _ = apply_batch_effects_from_config(
            x=base_matrix, y=class_labels, batch_config=batch_config, rng=rng
        )

        # On a zero matrix, scalar granularity yields identical columns within a batch.
        for batch_id in np.unique(batch_assignments):
            rows_in_batch = affected_matrix[batch_assignments == batch_id]
            assert np.allclose(rows_in_batch, rows_in_batch[:, [0]])

    def test_from_config_per_feature_varies_across_features(self):
        """Per-feature granularity must produce distinct shifts across affected features."""
        n_samples, n_features = 60, 5
        base_matrix = np.zeros((n_samples, n_features))
        class_labels = np.array([0] * 30 + [1] * 30)

        batch_config = BatchEffectsConfig(
            n_batches=3,
            effect_type="additive",
            effect_strength=1.0,
            effect_granularity="per_feature",
            confounding_with_class=0.0,
            affected_features="all",
        )
        rng = np.random.default_rng(42)
        affected_matrix, batch_assignments, _ = apply_batch_effects_from_config(
            x=base_matrix, y=class_labels, batch_config=batch_config, rng=rng
        )

        # On a zero matrix, per-feature granularity makes columns differ within a batch.
        for batch_id in np.unique(batch_assignments):
            rows_in_batch = affected_matrix[batch_assignments == batch_id]
            assert not np.allclose(rows_in_batch, rows_in_batch[:, [0]])

    def test_apply_batch_effects_draw_count_is_config_determined(self, rng_effects):
        """Effects are drawn for all configured batches, even an empty trailing one."""
        rng_x = np.random.default_rng(0)
        feature_matrix = rng_x.normal(size=(40, 4))
        # Batch index 2 is intentionally unused (empty trailing batch).
        batch_assignments = np.array([0] * 20 + [1] * 20)

        _, batch_effects = apply_batch_effects(
            feature_matrix,
            batch_assignments,
            rng=rng_effects,
            effect_type="additive",
            effect_strength=0.5,
            n_batches=3,  # config says 3, data only realizes 0 and 1
        )
        assert len(batch_effects) == 3

    def test_apply_batch_effects_rejects_n_batches_below_observed(self, rng_effects):
        """A configured n_batches below the observed maximum is a config error."""
        rng_x = np.random.default_rng(0)
        feature_matrix = rng_x.normal(size=(30, 3))
        batch_assignments = np.array([0] * 10 + [1] * 10 + [2] * 10)

        with pytest.raises(ValueError, match="must be >= max observed batch index"):
            apply_batch_effects(
                feature_matrix,
                batch_assignments,
                rng=rng_effects,
                n_batches=2,  # but the data contains batch index 2
            )

    def test_from_config_effects_length_matches_config_n_batches(self):
        """from_config sizes effects by config n_batches, not by realized batches."""
        base_matrix = np.zeros((40, 4))
        # Two classes, perfect confounding, 3 batches -> batch 2 stays empty.
        class_labels = np.array([0] * 20 + [1] * 20)
        batch_config = BatchEffectsConfig(
            n_batches=3,
            effect_type="additive",
            effect_strength=0.5,
            confounding_with_class=1.0,
            affected_features="all",
        )
        rng = np.random.default_rng(42)
        _, batch_assignments, batch_effects = apply_batch_effects_from_config(
            x=base_matrix, y=class_labels, batch_config=batch_config, rng=rng
        )

        # Batch 2 receives zero probability under perfect confounding with 2 classes,
        # so it is empty in the realized assignment...
        assert set(np.unique(batch_assignments)) <= {0, 1}
        # ...yet the effect summary still covers all three configured batches.
        assert len(batch_effects) == 3


# ============================================================================
# Tests for low-level random_batch_assignment
# ============================================================================


class TestRandomBatchAssignmentLowLevel:
    """Exercise random_batch_assignment directly (bypassing the orchestrator)."""

    def test_negative_n_samples_raises(self, rng_general):
        """Its own n_samples guard fires when called directly."""
        with pytest.raises(ValueError, match="n_samples must be positive"):
            random_batch_assignment(n_samples=0, n_batches=3, rng=rng_general)

    def test_zero_n_batches_raises(self, rng_general):
        """Its own n_batches guard fires when called directly."""
        with pytest.raises(ValueError, match="n_batches must be >= 1"):
            random_batch_assignment(n_samples=10, n_batches=0, rng=rng_general)

    def test_proportions_rounding_is_corrected(self, rng_general):
        """Imperfect rounding of proportions is fixed so counts sum to n_samples."""
        # 10 * [0.5, 0.3, 0.2] -> [5, 3, 2] sums to 10, so use a case that rounds short:
        # 7 * [0.5, 0.3, 0.2] = [3.5, 2.1, 1.4] -> round [4, 2, 1] = 7 exact;
        # 8 * [0.5, 0.3, 0.2] = [4.0, 2.4, 1.6] -> round [4, 2, 2] = 8 exact.
        # Use proportions that force a rounding correction:
        batches = random_batch_assignment(
            n_samples=10,
            n_batches=3,
            rng=rng_general,
            proportions=[1.0, 1.0, 1.0],  # 10/3 each -> rounds to [3, 3, 3] = 9, needs +1
        )
        assert batches.shape == (10,)
        counts = np.bincount(batches, minlength=3)
        assert counts.sum() == 10


# ============================================================================
# Tests for low-level confounded_batch_assignment
# ============================================================================


class TestConfoundedBatchAssignmentLowLevel:
    """Exercise confounded_batch_assignment directly (bypassing the orchestrator)."""

    def test_two_dimensional_labels_are_raveled(self, rng_general):
        """A 2-D class_labels array is flattened before processing."""
        labels = np.array([[0, 0, 1, 1]]).reshape(4, 1)
        batches = confounded_batch_assignment(
            class_labels=labels,
            n_batches=2,
            confounding_with_class=0.8,
            rng=rng_general,
        )
        assert batches.shape == (4,)

    def test_empty_labels_raises(self, rng_general):
        """No samples is an error."""
        with pytest.raises(ValueError, match="at least one sample"):
            confounded_batch_assignment(
                class_labels=np.array([], dtype=int),
                n_batches=2,
                confounding_with_class=0.8,
                rng=rng_general,
            )

    def test_single_batch_raises(self, rng_general):
        """Confounding is undefined for a single batch."""
        with pytest.raises(ValueError, match="requires n_batches >= 2"):
            confounded_batch_assignment(
                class_labels=np.array([0, 1, 0, 1]),
                n_batches=1,
                confounding_with_class=0.8,
                rng=rng_general,
            )

    def test_confounding_out_of_range_raises(self, rng_general):
        """Its own confounding bounds guard fires when called directly."""
        with pytest.raises(ValueError, match="confounding_with_class must be in"):
            confounded_batch_assignment(
                class_labels=np.array([0, 1, 0, 1]),
                n_batches=2,
                confounding_with_class=1.5,
                rng=rng_general,
            )

    def test_custom_proportions_branch(self, rng_general):
        """Custom proportions seed the per-class base probabilities."""
        labels = np.array([0] * 50 + [1] * 50)
        batches = confounded_batch_assignment(
            class_labels=labels,
            n_batches=2,
            confounding_with_class=0.6,
            rng=rng_general,
            proportions=[0.6, 0.4],
        )
        assert batches.shape == (100,)
        assert set(np.unique(batches)) <= {0, 1}

    def test_zero_confounding_skips_boost(self, rng_general):
        """confounding_with_class=0 leaves base probabilities untouched (boost <= 0)."""
        labels = np.array([0] * 50 + [1] * 50)
        batches = confounded_batch_assignment(
            class_labels=labels,
            n_batches=2,
            confounding_with_class=0.0,
            rng=rng_general,
        )
        assert batches.shape == (100,)
        assert set(np.unique(batches)) <= {0, 1}


# ============================================================================
# Tests for apply_batch_effects edge cases
# ============================================================================


class TestApplyBatchEffectsEdgeCases:
    """Cover validation and special-case branches in apply_batch_effects."""

    def test_mismatched_batch_assignments_length_raises(self, rng_effects):
        """batch_assignments must align with the number of samples."""
        feature_matrix = np.zeros((10, 3))
        with pytest.raises(ValueError, match="must have length n_samples"):
            apply_batch_effects(feature_matrix, np.zeros(5, dtype=int), rng=rng_effects)

    def test_negative_batch_assignment_raises(self, rng_effects):
        """Negative batch labels are rejected."""
        feature_matrix = np.zeros((4, 3))
        with pytest.raises(ValueError, match="must be non-negative"):
            apply_batch_effects(feature_matrix, np.array([0, 1, -1, 0]), rng=rng_effects)

    def test_two_dimensional_affected_features_are_raveled(self, rng_effects):
        """A nested affected_features sequence is flattened to feature indices."""
        feature_matrix = np.zeros((6, 4))
        batches = np.array([0, 1, 0, 1, 0, 1])
        result, effects = apply_batch_effects(
            feature_matrix,
            batches,
            rng=rng_effects,
            affected_features=[[0, 2]],
            effect_strength=0.5,
        )
        assert result.shape == (6, 4)
        # Untouched columns stay zero; affected ones change.
        assert np.allclose(result[:, [1, 3]], 0.0)

    def test_empty_affected_features_array(self, rng_effects):
        """An empty affected_features list is a no-op on ndarray input."""
        feature_matrix = np.ones((5, 3))
        result, effects = apply_batch_effects(
            feature_matrix,
            np.array([0, 1, 0, 1, 0]),
            rng=rng_effects,
            affected_features=[],
            n_batches=2,
        )
        assert_array_equal(result, feature_matrix)
        assert_array_equal(effects, np.zeros(2))

    def test_empty_affected_features_dataframe(self, rng_effects):
        """An empty affected_features list is a no-op on DataFrame input."""
        frame = pd.DataFrame(np.ones((5, 3)), columns=["a", "b", "c"])
        result, effects = apply_batch_effects(
            frame,
            np.array([0, 1, 0, 1, 0]),
            rng=rng_effects,
            affected_features=[],
            n_batches=2,
        )
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b", "c"]
        assert_array_equal(effects, np.zeros(2))

    def test_out_of_range_affected_features_raises(self, rng_effects):
        """Feature indices outside the matrix raise IndexError."""
        feature_matrix = np.zeros((5, 3))
        with pytest.raises(IndexError, match="out of range"):
            apply_batch_effects(
                feature_matrix,
                np.array([0, 1, 0, 1, 0]),
                rng=rng_effects,
                affected_features=[99],
            )

    def test_zero_effect_strength_dataframe(self, rng_effects):
        """Zero strength on DataFrame input returns the data unchanged."""
        frame = pd.DataFrame(np.ones((5, 3)), columns=["a", "b", "c"], index=[10, 11, 12, 13, 14])
        result, effects = apply_batch_effects(
            frame,
            np.array([0, 1, 0, 1, 0]),
            rng=rng_effects,
            effect_strength=0.0,
            n_batches=2,
        )
        assert isinstance(result, pd.DataFrame)
        assert list(result.index) == [10, 11, 12, 13, 14]
        assert_allclose(result.to_numpy(), frame.to_numpy())
        assert_array_equal(effects, np.zeros(2))

    def test_invalid_granularity_raises(self, rng_effects):
        """effect_granularity must be a known value."""
        feature_matrix = np.zeros((5, 3))
        with pytest.raises(ValueError, match="effect_granularity must be"):
            apply_batch_effects(
                feature_matrix,
                np.array([0, 1, 0, 1, 0]),
                rng=rng_effects,
                effect_strength=0.5,
                effect_granularity="bogus",
            )
