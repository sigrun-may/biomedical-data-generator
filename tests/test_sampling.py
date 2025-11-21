# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tests for sampling utilities."""

import numpy as np
import pytest

from biomedical_data_generator.utils.sampling import sample_2d_array


def test_sample_2d_array_normal():
    """Test sampling from normal distribution."""
    rng = np.random.default_rng(42)
    samples = sample_2d_array(
        distribution="normal",
        params={"loc": 0.0, "scale": 1.0},
        rng=rng,
        size=(100, 5),
    )

    assert samples.shape == (100, 5)
    assert np.abs(samples.mean()) < 0.3  # Should be close to 0
    assert 0.7 < samples.std() < 1.3  # Should be close to 1


def test_sample_2d_array_uniform():
    """Test sampling from uniform distribution."""
    rng = np.random.default_rng(42)
    samples = sample_2d_array(
        distribution="uniform",
        params={"low": 0.0, "high": 10.0},
        rng=rng,
        size=(100, 5),
    )

    assert samples.shape == (100, 5)
    assert samples.min() >= 0.0
    assert samples.max() <= 10.0
    assert 4.0 < samples.mean() < 6.0  # Should be around 5


def test_sample_2d_array_exponential():
    """Test sampling from exponential distribution."""
    rng = np.random.default_rng(42)
    samples = sample_2d_array(
        distribution="exponential",
        params={"scale": 2.0},
        rng=rng,
        size=(1000, 3),
    )

    assert samples.shape == (1000, 3)
    assert samples.min() >= 0.0
    assert 1.5 < samples.mean() < 2.5  # Mean should be close to scale


def test_sample_2d_array_laplace():
    """Test sampling from Laplace distribution."""
    rng = np.random.default_rng(42)
    samples = sample_2d_array(
        distribution="laplace",
        params={"loc": 5.0, "scale": 1.0},
        rng=rng,
        size=(100, 5),
    )

    assert samples.shape == (100, 5)
    assert 4.0 < samples.mean() < 6.0


def test_sample_2d_array_lognormal():
    """Test sampling from lognormal distribution."""
    rng = np.random.default_rng(42)
    samples = sample_2d_array(
        distribution="lognormal",
        params={"mean": 0.0, "sigma": 1.0},
        rng=rng,
        size=(100, 5),
    )

    assert samples.shape == (100, 5)
    assert samples.min() > 0.0  # Lognormal is always positive


def test_sample_2d_array_exp_normal():
    """Test sampling from exp_normal distribution."""
    rng = np.random.default_rng(42)
    samples = sample_2d_array(
        distribution="exp_normal",
        params={"loc": 0.0, "scale": 1.0},
        rng=rng,
        size=(100, 5),
    )

    assert samples.shape == (100, 5)
    assert samples.min() > 0.0  # exp() is always positive


def test_sample_2d_array_empty_params():
    """Test sampling with empty params (use defaults)."""
    rng = np.random.default_rng(42)
    samples = sample_2d_array(
        distribution="normal",
        params={},
        rng=rng,
        size=(10, 3),
    )

    assert samples.shape == (10, 3)


def test_sample_2d_array_unsupported_distribution_raises():
    """Test that unsupported distribution raises ValueError."""
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError, match="Unsupported distribution"):
        sample_2d_array(
            distribution="nonexistent_distribution",
            params={},
            rng=rng,
            size=(10, 3),
        )


def test_sample_2d_array_different_sizes():
    """Test sampling with different matrix sizes."""
    rng = np.random.default_rng(42)

    # Small matrix
    small = sample_2d_array("normal", {}, rng, size=(5, 2))
    assert small.shape == (5, 2)

    # Large matrix
    large = sample_2d_array("normal", {}, rng, size=(1000, 100))
    assert large.shape == (1000, 100)

    # Single column
    single_col = sample_2d_array("normal", {}, rng, size=(10, 1))
    assert single_col.shape == (10, 1)


def test_sample_2d_array_reproducibility():
    """Test that same seed produces same results."""
    rng1 = np.random.default_rng(123)
    samples1 = sample_2d_array("normal", {}, rng1, size=(10, 5))

    rng2 = np.random.default_rng(123)
    samples2 = sample_2d_array("normal", {}, rng2, size=(10, 5))

    np.testing.assert_array_equal(samples1, samples2)


def test_sample_2d_array_exp_normal_is_positive():
    """Test that exp_normal produces only positive values."""
    rng = np.random.default_rng(42)

    # Even with negative loc, exp should make all values positive
    samples = sample_2d_array(
        distribution="exp_normal",
        params={"loc": -10.0, "scale": 1.0},
        rng=rng,
        size=(100, 5),
    )

    assert np.all(samples > 0.0)
