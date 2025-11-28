# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tests for correlated biomarker cluster generation."""

import numpy as np
import pytest

from biomedical_data_generator.features import correlated as corr
from biomedical_data_generator.utils import correlation_tools

# -------------------------
# build_correlation_matrix
# -------------------------


def test_build_corr_equicorrelated_exact_values():
    p, correlation = 5, 0.6
    R = corr.build_correlation_matrix(p, correlation, "equicorrelated")
    assert R.shape == (p, p)
    assert R.dtype == np.float64
    # Diagonal exactly 1
    assert np.allclose(np.diag(R), 1.0)
    # Off-diagonals exactly correlation for equicorrelated
    mask = ~np.eye(p, dtype=bool)
    assert np.allclose(R[mask], correlation)


def test_build_corr_toeplitz_exact_values():
    p, correlation = 6, 0.4
    R = corr.build_correlation_matrix(p, correlation, "toeplitz")
    # Diagonal exactly 1
    assert np.allclose(np.diag(R), 1.0)
    # Check a few Toeplitz powers
    for d in (1, 2, 3):
        vals = np.diag(R, k=d)
        assert np.allclose(vals, correlation**d, atol=1e-12)


def test_build_corr_invalid_correlation_equicorr_raises():
    p = 4
    lower = -1.0 / (p - 1)  # = -1/3
    with pytest.raises(ValueError):
        corr.build_correlation_matrix(p, lower, "equicorrelated")
    with pytest.raises(ValueError):
        corr.build_correlation_matrix(p, 1.0, "equicorrelated")


def test_build_corr_invalid_correlation_toeplitz_raises():
    with pytest.raises(ValueError):
        corr.build_correlation_matrix(5, -1.0, "toeplitz")
    with pytest.raises(ValueError):
        corr.build_correlation_matrix(5, 1.0, "toeplitz")


def test_build_corr_unknown_structure_raises():
    with pytest.raises(ValueError):
        corr.build_correlation_matrix(3, 0.5, "not-a-structure")


# --------------
# offdiag_metrics
# --------------


def test_offdiag_mean_behavior():
    # p==1 => defined as 1.0
    assert correlation_tools.compute_correlation_metrics(np.array([[1.0]]))["mean_offdiag"] == 1.0
    # equicorrelated matrix => mean equals correlation
    R = corr.build_correlation_matrix(5, 0.3, "equicorrelated")
    assert np.isclose(correlation_tools.compute_correlation_metrics(R)["mean_offdiag"], 0.3)


# --------------
# compute_correlation_metrics
# --------------


def test_compute_correlation_metrics_single_feature():
    """Single feature edge case returns expected values."""
    # p==1 => defined as 1.0
    metrics = correlation_tools.compute_correlation_metrics(np.array([[1.0]]))
    assert metrics["mean_offdiag"] == 1.0
    assert metrics["n_offdiag"] == 0


def test_compute_correlation_metrics_equicorrelated():
    """Equicorrelated matrix => mean_offdiag equals correlation."""
    R = corr.build_correlation_matrix(5, 0.3, "equicorrelated")
    metrics = correlation_tools.compute_correlation_metrics(R)
    assert np.isclose(metrics["mean_offdiag"], 0.3)


# ----------------
# _cholesky_with_jitter
# ----------------


def test_cholesky_with_jitter_handles_near_singular():
    p = 6
    # Make an almost-singular equicorrelated matrix and push it slightly non-PD
    R = corr.build_correlation_matrix(p, 1.0 - 1e-12, "equicorrelated")
    R_bad = R - 1e-8 * np.eye(p)
    # Should succeed by escalating jitter; no exception
    L = corr._cholesky_with_jitter(R_bad, initial_jitter=1e-12, growth=10.0, max_tries=8)
    assert L.shape == (p, p)
    # Lower-triangular check (within numerical tolerance)
    assert np.allclose(L, np.tril(L))


# -------------
# sample_cluster
# -------------


def test_sample_cluster_global_equicorrelated_matches_mean():
    n, p, correlation = 500, 6, 0.6
    rng = np.random.default_rng(0)
    x = corr.sample_correlated_data(
        n_samples=n, n_features=p, rng=rng, structure="equicorrelated", correlation=correlation
    )
    c_emp = np.corrcoef(x, rowvar=False).astype(np.float64)
    mean_off = correlation_tools.compute_correlation_metrics(c_emp)["mean_offdiag"]
    assert np.isfinite(mean_off)
    # Sampling noise: allow small tolerance
    assert abs(mean_off - correlation) <= 0.06


def test_sample_cluster_global_toeplitz_lag_decay():
    n, p, correlation = 800, 6, 0.35
    rng = np.random.default_rng(1)
    X = corr.sample_correlated_data(n_samples=n, n_features=p, rng=rng, structure="toeplitz", correlation=correlation)
    C_emp = np.corrcoef(X, rowvar=False).astype(np.float64)
    # First and second off-diagonals should be close to correlation and correlation**2
    lag1 = np.mean([C_emp[i, i + 1] for i in range(p - 1)])
    lag2 = np.mean([C_emp[i, i + 2] for i in range(p - 2)])
    assert abs(lag1 - correlation) <= 0.06
    assert abs(lag2 - correlation**2) <= 0.06


def test_sample_cluster_errors():
    rng = np.random.default_rng(0)
    # Missing correlation in global mode
    with pytest.raises(TypeError):
        corr.sample_correlated_data(n_samples=10, n_features=3, rng=rng)
    # Missing class-specific correlation
    with pytest.raises(TypeError):
        corr.sample_correlated_data(
            n_samples=10,
            n_features=3,
            rng=rng,
            structure="equicorrelated",
            correlation={0: 0.5},
        )
    # Invalid structure
    with pytest.raises(ValueError):
        corr.sample_correlated_data(
            n_samples=10,
            n_features=3,
            rng=rng,
            structure="invalid-structure",
            correlation=0.5,
        )


# -------------------------
# find_seed_for_correlation
# -------------------------
def test_find_seed_for_correlation_tol_mode():
    seed, meta = correlation_tools.find_seed_for_correlation(
        n_samples=200,
        n_cluster_features=4,
        correlation=0.5,
        structure="equicorrelated",
        tolerance=0.03,
        start_seed=0,
        max_tries=50,
    )
    assert isinstance(seed, int)
    assert meta["accepted"] is True
    assert abs(meta["mean_offdiag"] - 0.5) <= 0.03
    assert int(meta["tries"]) >= 1


def test_find_seed_for_correlation_impossible_threshold_raises():
    # Make acceptance impossible with few tries => should raise RuntimeError (once bug above is fixed)
    with pytest.raises(RuntimeError):
        correlation_tools.find_seed_for_correlation(
            n_samples=80,
            n_cluster_features=5,
            correlation=0.2,
            structure="toeplitz",
            metric="min_offdiag",
            threshold=0.99,  # unrealistic
            op=">=",
            tolerance=None,
            start_seed=0,
            max_tries=3,
            return_best_on_fail=False,
        )


def test_find_seed_for_correlation_best_on_fail_fallback():
    """With return_best_on_fail=True, returns best seed instead of raising."""
    # FIXED: Added new test for the fallback behavior
    seed, meta = correlation_tools.find_seed_for_correlation(
        n_samples=80,
        n_cluster_features=5,
        correlation=0.2,
        structure="toeplitz",
        metric="min_offdiag",
        threshold=0.99,  # unrealistic
        op=">=",
        tolerance=None,
        start_seed=0,
        max_tries=3,
        return_best_on_fail=True,  # Should return best instead of raising
    )
    assert isinstance(seed, int)
    assert meta["accepted"] is False  # Didn't meet criteria
    assert meta["tries"] == 3
    assert "min_offdiag" in meta


def test_find_seed_for_correlation_threshold_mode():
    """Threshold mode accepts when metric satisfies threshold."""
    seed, meta = correlation_tools.find_seed_for_correlation(
        n_samples=300,
        n_cluster_features=6,
        correlation=0.65,
        structure="equicorrelated",
        metric="min_offdiag",
        threshold=0.50,  # Achievable threshold
        op=">=",
        tolerance=None,
        start_seed=0,
        max_tries=100,
    )
    assert isinstance(seed, int)
    assert meta["accepted"] is True
    assert meta["min_offdiag"] >= 0.50


def test_find_seed_for_correlation_return_matrix():
    """return_matrix=True includes correlation matrix in metadata."""
    seed, meta = correlation_tools.find_seed_for_correlation(
        n_samples=100,
        n_cluster_features=4,
        correlation=0.6,
        structure="equicorrelated",
        tolerance=0.05,
        start_seed=0,
        max_tries=20,
        return_matrix=True,
    )
    assert "corr_matrix" in meta
    assert isinstance(meta["corr_matrix"], np.ndarray)
    assert meta["corr_matrix"].shape == (4, 4)


def test_find_seed_for_correlation_p_gt_n_warning():
    """enforce_p_le_n_in_tolerance warns when p > n."""
    seed, meta = correlation_tools.find_seed_for_correlation(
        n_samples=50,
        n_cluster_features=100,  # p > n
        correlation=0.5,
        structure="equicorrelated",
        tolerance=0.03,
        start_seed=0,
        max_tries=10,
        enforce_p_le_n_in_tolerance=True,
    )
    # Should have triggered warning and auto-rejected all attempts
    assert meta["p_gt_n_tolerance_warning"] is True
    assert meta["accepted"] is False  # Can't accept with p > n enforcement


# -------------------------
# find_best_seed_for_correlation
# -------------------------
def test_find_best_seed_for_correlation_returns_best():
    """Best-of-N scanner returns seed with smallest deviation."""
    seed, metrics = correlation_tools.find_best_seed_for_correlation(
        max_tries=20,
        n_samples=200,
        n_cluster_features=5,
        correlation=0.65,
        structure="equicorrelated",
        start_seed=0,
    )

    assert isinstance(seed, int)
    assert 0 <= seed < 20
    assert "delta_offdiag" in metrics
    assert "mean_offdiag" in metrics
    # Delta should be reasonably small (best of 20 attempts)
    assert metrics["delta_offdiag"] < 0.1


# -------------------------
# assess_correlation_quality
# -------------------------


def test_assess_correlation_quality():
    """Quality assessment computes all metrics and checks tolerance."""
    rng = np.random.default_rng(42)
    X = corr.sample_correlated_data(300, 6, correlation=0.65, rng=rng, structure="equicorrelated")

    quality = correlation_tools.assess_correlation_quality(X, correlation_target=0.65, tolerance=0.03)

    assert "mean_offdiag" in quality
    assert "deviation_offdiag" in quality
    assert "within_tolerance" in quality
    assert "structure" in quality
    assert quality["target"] == 0.65
    assert isinstance(quality["within_tolerance"], bool)
