# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tests for informative features generation."""

import numpy as np
from scipy import stats

from biomedical_data_generator import ClassConfig, DatasetConfig
from biomedical_data_generator.features.informative import (
    _build_class_labels,
    _class_offsets_from_sep,
    generate_informative_features,
)


def _scipy_dist_and_args(dist_name: str, params: dict):
    """Map generator distribution + params -> (scipy_name, args_tuple)."""
    if dist_name == "normal":
        loc = float(params.get("loc", 0.0))
        scale = float(params.get("scale", 1.0))
        return "norm", (loc, scale)
    if dist_name == "uniform":
        low = float(params["low"])
        high = float(params["high"])
        return "uniform", (low, high - low)
    if dist_name == "laplace":
        loc = float(params.get("loc", 0.0))
        scale = float(params.get("scale", 1.0))
        return "laplace", (loc, scale)
    if dist_name == "exponential":
        scale = float(params.get("scale", 1.0))
        return "expon", (0.0, scale)
    if dist_name == "lognormal":
        mean = float(params.get("mean", 0.0))
        sigma = float(params.get("sigma", 1.0))
        # scipy.stats.lognorm: shape = sigma, loc, scale = exp(mean)
        return "lognorm", (sigma, 0.0, float(np.exp(mean)))
    if dist_name == "exp_normal":
        loc = float(params.get("loc", 0.0))
        scale = float(params.get("scale", 1.0))
        # exp_normal == exp(normal(loc, scale)) -> same mapping as lognormal
        return "lognorm", (scale, 0.0, float(np.exp(loc)))
    raise ValueError(f"Unknown distribution: {dist_name}")


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


def test_generate_informative_features_different_distributions_statistical():
    """Statistical check of per-class sampling using one-sample KS tests.

    Uses large sample sizes and class_sep=0 to avoid mean shifts so the
    sampled values can be tested directly against the nominal distributions.
    Requires SciPy; skipped if not available.
    """
    # Large sample sizes for good test power
    n_per_class = 5000

    cfg = DatasetConfig(
        n_informative=3,
        n_noise=0,
        class_configs=[
            ClassConfig(
                n_samples=n_per_class,
                class_distribution="normal",
                class_distribution_params={"loc": 0.0, "scale": 1.0},
            ),
            ClassConfig(
                n_samples=n_per_class,
                class_distribution="lognormal",
                class_distribution_params={"mean": 0.0, "sigma": 1.0},
            ),
        ],
        class_sep=[1.5],  # disable shifting so we test the raw sampled distributions
        random_state=42,
    )

    rng = np.random.default_rng(42)
    X, y = generate_informative_features(cfg, rng)

    # Undo class-wise offsets
    offsets = _class_offsets_from_sep(cfg.class_sep)
    assert offsets.size == len(cfg.class_configs)

    # Pick the first informative column for each class
    x_class0 = X[y == 0, 0] - offsets[0]
    x_class1 = X[y == 1, 0] - offsets[1]

    alpha = 0.01

    # KS test for normal(0,1)
    stat_norm, p_norm = stats.kstest(x_class0, "norm", args=(0.0, 1.0))
    assert p_norm > alpha, f"Normal KS test failed (p={p_norm:.5g})"

    # KS test for lognormal(0,1)
    stat_unif, p_lognorm = stats.kstest(x_class1, "lognorm", args=(1.0, 0.0, np.exp(0.0)))
    assert p_lognorm > alpha, f"Lognormal KS test failed (p={p_lognorm:.5g})"


def test_generate_informative_features_all_supported_distributions_statistical():
    """KS tests for all supported distributions (class_sep=0 to avoid shifts)."""
    n_per_class = 5000
    alpha = 0.01

    dist_cases = {
        "normal": {"loc": 0.0, "scale": 1.0},
        "uniform": {"low": 5.0, "high": 10.0},
        "laplace": {"loc": 0.0, "scale": 1.0},
        "exponential": {"scale": 2.0},
        "lognormal": {"mean": 0.0, "sigma": 0.5},
        "exp_normal": {"loc": 0.0, "scale": 0.5},
    }

    for dist_name, params in dist_cases.items():
        cfg = DatasetConfig(
            n_informative=1,
            n_noise=0,
            class_configs=[
                ClassConfig(
                    n_samples=n_per_class,
                    class_distribution=dist_name,
                    class_distribution_params=params,
                ),
                ClassConfig(
                    n_samples=n_per_class,
                    class_distribution="normal",
                    class_distribution_params={"loc": 0.0, "scale": 1.0},
                ),
            ],
            class_sep=1.0,
            random_state=42,
        )

        rng = np.random.default_rng(42)
        X, y = generate_informative_features(cfg, rng)

        # Remove the applied class offsets to recover base samples before KS test
        offsets = _class_offsets_from_sep(cfg.class_sep)
        x_class0 = X[y == 0, 0] - offsets[0]

        scipy_name, args = _scipy_dist_and_args(dist_name, params)
        stat, pval = stats.kstest(x_class0, scipy_name, args=args)
        assert pval > alpha, f"{dist_name} KS test failed (p={pval:.5g})"


def test_generate_informative_features_shift_preserves_distribution_shape():
    """Ensure class-wise shifts are pure mean-offsets and do not change distribution shape."""
    n_per_class = 5000
    alpha = 0.01

    # Two identical classes sampled from standard normal, apply class_sep to shift means
    cfg = DatasetConfig(
        n_informative=1,
        n_noise=0,
        class_configs=[
            ClassConfig(
                n_samples=n_per_class, class_distribution="normal", class_distribution_params={"loc": 0, "scale": 1}
            ),
            ClassConfig(
                n_samples=n_per_class, class_distribution="normal", class_distribution_params={"loc": 0, "scale": 1}
            ),
        ],
        class_sep=0.8,  # non-zero to trigger shifts
        random_state=42,
    )

    rng = np.random.default_rng(42)
    X, y = generate_informative_features(cfg, rng)  # returns shifted features

    # Recompute the offsets used by shift_classes and remove them to recover base samples
    offsets = _class_offsets_from_sep(cfg.class_sep)

    # For each class, subtract the applied offset and KS-test against the original distribution
    for k in range(cfg.n_classes):
        xk = X[y == k, 0] - offsets[k]
        stat, pval = stats.kstest(xk, "norm", args=(0.0, 1.0))
        assert pval > alpha, f"class {k} distribution altered by shift (KS p={pval:.5g})"
