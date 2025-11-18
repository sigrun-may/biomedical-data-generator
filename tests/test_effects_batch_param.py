# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Parametrized tests for batch effects - comprehensive coverage.

These tests use pytest.mark.parametrize to test many combinations
of parameters efficiently.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from biomedical_data_generator.effects.batch import (
    apply_batch_effects,
    generate_batch_assignments,
)

# ============================================================================
# Parametrized tests for batch assignments
# ============================================================================


@pytest.mark.parametrize(
    "n_samples,n_batches",
    [
        (100, 2),
        (100, 3),
        (100, 5),
        (150, 10),
        (200, 4),
    ],
)
def test_assignment_sample_counts(n_samples, n_batches):
    """All samples should be assigned exactly once."""
    rng = np.random.default_rng(42)
    batches = generate_batch_assignments(n_samples, n_batches=n_batches, rng=rng)

    assert len(batches) == n_samples
    assert np.min(batches) == 0
    assert np.max(batches) == n_batches - 1
    assert np.sum(np.bincount(batches)) == n_samples


@pytest.mark.parametrize("confounding_with_class", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
def test_confounding_with_class_spectrum(confounding_with_class):
    """Test confounding across full strength spectrum."""
    y = np.array([0] * 100 + [1] * 100)
    rng = np.random.default_rng(42)
    batches = generate_batch_assignments(
        200,
        n_batches=2,
        class_labels=y,
        confounding_with_class=confounding_with_class,
        rng=rng,
    )

    # Count class 0 in batch 0
    class0_in_batch0 = np.sum((y == 0) & (batches == 0))

    # Redistribution formula (2 batches, equal base):
    # P(preferred) = 0.5 + 0.5 * strength
    #
    # strength | P(preferred) | Expected count | Range (with variance)
    # ---------|--------------|----------------|---------------------
    #   0.0    |     0.50     |      50        |    40-60
    #   0.2    |     0.60     |      60        |    52-68
    #   0.4    |     0.70     |      70        |    62-78
    #   0.6    |     0.80     |      80        |    72-88
    #   0.8    |     0.90     |      90        |    82-96
    #   1.0    |     1.00     |     100        |    95-100

    if confounding_with_class == 0.0:
        assert 40 < class0_in_batch0 < 60  # Random (~50%)
    elif confounding_with_class <= 0.3:
        assert 50 < class0_in_batch0 < 70  # Slight preference (~60%)
    elif confounding_with_class <= 0.6:
        assert 65 < class0_in_batch0 < 88  # Moderate preference (~75%)
    elif confounding_with_class < 1.0:
        assert class0_in_batch0 > 75  # Strong preference (>75%)
    else:  # == 1.0
        assert class0_in_batch0 >= 95  # Perfect/near-perfect (>=95%)


@pytest.mark.parametrize(
    "n_classes,n_batches",
    [
        (2, 2),
        (3, 3),
        (5, 3),  # More classes than batches
        (3, 5),  # More batches than classes
        (10, 4),
    ],
)
def test_class_batch_combinations(n_classes, n_batches):
    """Test various class/batch combinations."""
    samples_per_class = 20
    y = np.repeat(np.arange(n_classes), samples_per_class)

    rng = np.random.default_rng(42)
    batches = generate_batch_assignments(
        len(y),
        n_batches=n_batches,
        class_labels=y,
        confounding_with_class=0.5,
        rng=rng,
    )

    assert len(batches) == len(y)
    assert set(batches) <= set(range(n_batches))


@pytest.mark.parametrize(
    "proportions",
    [
        [0.5, 0.5],
        [0.7, 0.3],
        [0.3, 0.3, 0.4],
        [0.25, 0.25, 0.25, 0.25],
        [0.6, 0.2, 0.2],
    ],
)
def test_various_proportions(proportions):
    """Test different proportion specifications."""
    n_batches = len(proportions)
    n_samples = 200

    rng = np.random.default_rng(42)
    batches = generate_batch_assignments(
        n_samples,
        n_batches=n_batches,
        proportions=proportions,
        rng=rng,
    )

    counts = np.bincount(batches)
    observed_props = counts / n_samples
    expected_props = np.array(proportions)

    # Allow ±5% deviation due to rounding
    assert_allclose(observed_props, expected_props, atol=0.05)


# ============================================================================
# Parametrized tests for batch effects
# ============================================================================


@pytest.mark.parametrize("effect_type", ["additive", "multiplicative"])
@pytest.mark.parametrize("effect_strength", [0.0, 0.3, 0.5, 1.0, 2.0])
def test_effect_types_and_strengths(effect_type, effect_strength):
    """Test all combinations of effect types and strengths."""
    X = np.random.randn(100, 5)
    rng_assign = np.random.default_rng(42)
    batches = generate_batch_assignments(100, n_batches=3, rng=rng_assign)

    rng_effects = np.random.default_rng(42)
    X_batch, batch_effects = apply_batch_effects(
        X,
        batches,
        rng=rng_effects,
        effect_type=effect_type,
        effect_strength=effect_strength,
    )

    assert X_batch.shape == X.shape
    assert len(batch_effects) == 3

    # With zero strength, should be unchanged
    if effect_strength == 0.0:
        assert_allclose(X_batch, X, atol=1e-10)
        assert_allclose(batch_effects, 0.0, atol=1e-10)


@pytest.mark.parametrize("n_features", [5, 10, 20, 50])
@pytest.mark.parametrize("n_batches", [2, 4, 8])
def test_scalability(n_features, n_batches):
    """Test performance with different data dimensions."""
    n_samples = 200
    X = np.random.randn(n_samples, n_features)

    rng_assign = np.random.default_rng(42)
    batches = generate_batch_assignments(n_samples, n_batches=n_batches, rng=rng_assign)

    rng_effects = np.random.default_rng(43)
    X_batch, batch_effects = apply_batch_effects(
        X,
        batches,
        rng=rng_effects,
        effect_type="additive",
        effect_strength=0.5,
    )

    assert X_batch.shape == (n_samples, n_features)
    assert len(batch_effects) == n_batches


@pytest.mark.parametrize(
    "affected_features",
    [
        "all",
        [0],
        [0, 1],
        [0, 2, 4],
        list(range(5)),
    ],
)
def test_feature_selection_patterns(affected_features):
    """Test different feature selection patterns."""
    X = np.random.randn(80, 5)
    rng_assign = np.random.default_rng(42)
    batches = generate_batch_assignments(80, n_batches=2, rng=rng_assign)

    X_orig = X.copy()
    rng_effects = np.random.default_rng(42)
    X_batch, _ = apply_batch_effects(
        X,
        batches,
        rng=rng_effects,
        effect_type="additive",
        effect_strength=0.5,
        affected_features=affected_features,
    )

    if affected_features == "all":
        # All features should change
        for i in range(5):
            assert not np.allclose(X_orig[:, i], X_batch[:, i])
    else:
        # Only specified features should change
        for i in range(5):
            if i in affected_features:
                assert not np.allclose(X_orig[:, i], X_batch[:, i])
            else:
                assert_allclose(X_orig[:, i], X_batch[:, i])


@pytest.mark.parametrize("input_type", ["array", "dataframe"])
@pytest.mark.parametrize("effect_type", ["additive", "multiplicative"])
def test_input_output_consistency(input_type, effect_type):
    """Test that input type determines output type."""
    if input_type == "array":
        X = np.random.randn(50, 5)
    else:
        X = pd.DataFrame(np.random.randn(50, 5), columns=[f"f{i}" for i in range(5)])

    rng_assign = np.random.default_rng(42)
    batches = generate_batch_assignments(50, n_batches=2, rng=rng_assign)

    rng_effects = np.random.default_rng(43)
    X_batch, _ = apply_batch_effects(
        X,
        batches,
        rng=rng_effects,
        effect_type=effect_type,
        effect_strength=0.5,
    )

    if input_type == "array":
        assert isinstance(X_batch, np.ndarray)
    else:
        assert isinstance(X_batch, pd.DataFrame)
        assert list(X_batch.columns) == [f"f{i}" for i in range(5)]


# ============================================================================
# Statistical property tests
# ============================================================================
@pytest.mark.parametrize("n_batches", [2, 3, 5, 10])
def test_batch_variance_decomposition(n_batches):
    """Verify batch effects add variance in predictable way."""
    n_samples = 500
    X = np.random.randn(n_samples, 1)  # Single feature for clarity

    rng_assign = np.random.default_rng(42)
    batches = generate_batch_assignments(
        n_samples,
        n_batches=n_batches,
        rng=rng_assign,
    )

    var_original = X.var()

    rng_effects = np.random.default_rng(43)
    X_batch, batch_effects = apply_batch_effects(
        X,
        batches,
        rng=rng_effects,
        effect_type="additive",
        effect_strength=1.0,
    )

    var_with_batch = X_batch.var()

    # Variance should increase
    assert var_with_batch > var_original

    # Batch effects should nicht degeneriert sein (nicht alle exakt 0)
    # (Bei effect_strength=1.0 praktisch sicher, aber explizit getestet.)
    assert not np.allclose(batch_effects, 0.0, atol=1e-12)


@pytest.mark.parametrize("effect_strength", [0.1, 0.5, 1.0, 2.0])
def test_effect_strength_correlation(effect_strength):
    """Higher effect_strength should produce more divergence."""
    X = np.random.randn(200, 1)

    rng_assign = np.random.default_rng(42)
    batches = generate_batch_assignments(200, n_batches=3, rng=rng_assign)

    rng_effects = np.random.default_rng(43)
    X_batch, batch_effects = apply_batch_effects(
        X,
        batches,
        rng=rng_effects,
        effect_type="additive",
        effect_strength=effect_strength,
    )

    # Batch effect std should scale with effect_strength
    effect_std = np.std(batch_effects)
    # Should be roughly proportional (within 2–3x due to sampling)
    assert 0.3 * effect_strength < effect_std < 3.0 * effect_strength


# ============================================================================
# Edge case tests
# ============================================================================


@pytest.mark.parametrize("n_samples", [1, 2, 5, 10])
def test_very_small_samples(n_samples):
    """Handle very small sample sizes gracefully."""
    n_batches = min(n_samples, 2)

    rng_assign = np.random.default_rng(42)
    batches = generate_batch_assignments(n_samples, n_batches=n_batches, rng=rng_assign)

    assert len(batches) == n_samples

    X = np.random.randn(n_samples, 3)
    rng_effects = np.random.default_rng(43)
    X_batch, _ = apply_batch_effects(X, batches, rng=rng_effects)

    assert X_batch.shape == X.shape


@pytest.mark.parametrize("n_batches", [1, 2, 5, 20, 50])
def test_various_batch_counts(n_batches):
    """Test from single batch to many batches."""
    n_samples = 100

    rng_assign = np.random.default_rng(42)
    batches = generate_batch_assignments(n_samples, n_batches=n_batches, rng=rng_assign)

    assert len(batches) == n_samples
    assert len(np.unique(batches)) <= n_batches

    X = np.random.randn(n_samples, 5)
    rng_effects = np.random.default_rng(43)
    X_batch, batch_effects = apply_batch_effects(X, batches, rng=rng_effects)

    assert X_batch.shape == (n_samples, 5)
    assert len(batch_effects) == n_batches


# ============================================================================
# Reproducibility tests
# ============================================================================


@pytest.mark.parametrize("seed", [0, 42, 123, 999, 2025])
def test_seed_reproducibility(seed):
    """Same seed should always produce same results."""
    y = np.array([0] * 50 + [1] * 50)

    rng1 = np.random.default_rng(seed)
    batches1 = generate_batch_assignments(
        100,
        n_batches=3,
        class_labels=y,
        confounding_with_class=0.5,
        rng=rng1,
    )

    rng2 = np.random.default_rng(seed)
    batches2 = generate_batch_assignments(
        100,
        n_batches=3,
        class_labels=y,
        confounding_with_class=0.5,
        rng=rng2,
    )

    assert_allclose(batches1, batches2)

    X = np.random.randn(100, 5)

    rng3 = np.random.default_rng(seed)
    X_batch1, effects1 = apply_batch_effects(X, batches1, rng=rng3)

    rng4 = np.random.default_rng(seed)
    X_batch2, effects2 = apply_batch_effects(X, batches2, rng=rng4)

    assert_allclose(X_batch1, X_batch2)
    assert_allclose(effects1, effects2)
