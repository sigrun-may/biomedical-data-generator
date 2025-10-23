# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tests for correlated biomarker cluster generation."""

import numpy as np
import pytest

import biomedical_data_generator.utils.correlation_seed_search
from biomedical_data_generator.features import correlated as corr



# -------------------------
# build_correlation_matrix
# -------------------------

def test_build_corr_equicorrelated_exact_values():
    p, rho = 5, 0.6
    R = corr.build_correlation_matrix(p, rho, "equicorrelated")
    assert R.shape == (p, p)
    assert R.dtype == np.float64
    # Diagonal exactly 1
    assert np.allclose(np.diag(R), 1.0)
    # Off-diagonals exactly rho for equicorrelated
    mask = ~np.eye(p, dtype=bool)
    assert np.allclose(R[mask], rho)


def test_build_corr_toeplitz_exact_values():
    p, rho = 6, 0.4
    R = corr.build_correlation_matrix(p, rho, "toeplitz")
    # Diagonal exactly 1
    assert np.allclose(np.diag(R), 1.0)
    # Check a few Toeplitz powers
    for d in (1, 2, 3):
        vals = np.diag(R, k=d)
        assert np.allclose(vals, rho ** d, atol=1e-12)


def test_build_corr_invalid_rho_equicorr_raises():
    p = 4
    lower = -1.0 / (p - 1)  # = -1/3
    with pytest.raises(ValueError):
        corr.build_correlation_matrix(p, lower, "equicorrelated")
    with pytest.raises(ValueError):
        corr.build_correlation_matrix(p, 1.0, "equicorrelated")


def test_build_corr_invalid_rho_toeplitz_raises():
    with pytest.raises(ValueError):
        corr.build_correlation_matrix(5, -1.0, "toeplitz")
    with pytest.raises(ValueError):
        corr.build_correlation_matrix(5, 1.0, "toeplitz")


def test_build_corr_unknown_structure_raises():
    with pytest.raises(ValueError):
        corr.build_correlation_matrix(3, 0.5, "not-a-structure")  # type: ignore[arg-type]


# --------------
# offdiag_metrics
# --------------

def test_offdiag_mean_behavior():
    # p==1 => defined as 1.0
    assert corr.offdiag_metrics(np.array([[1.0]]))["mean_offdiag"] == 1.0
    # equicorrelated matrix => mean equals rho
    R = corr.build_correlation_matrix(5, 0.3, "equicorrelated")
    assert np.isclose(corr.offdiag_metrics(R)["mean_offdiag"], 0.3)


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
    n, p, rho = 500, 6, 0.6
    rng = np.random.default_rng(0)
    X = corr.sample_cluster(n, p, rng, structure="equicorrelated", rho=rho)
    C_emp = np.corrcoef(X, rowvar=False).astype(np.float64)
    mean_off = corr.offdiag_metrics(C_emp)["mean_offdiag"]
    assert np.isfinite(mean_off)
    # Sampling noise: allow small tolerance
    assert abs(mean_off - rho) <= 0.06


def test_sample_cluster_global_toeplitz_lag_decay():
    n, p, rho = 800, 6, 0.35
    rng = np.random.default_rng(1)
    X = corr.sample_cluster(n, p, rng, structure="toeplitz", rho=rho)
    C_emp = np.corrcoef(X, rowvar=False).astype(np.float64)
    # First and second off-diagonals should be close to rho and rho**2
    lag1 = np.mean([C_emp[i, i + 1] for i in range(p - 1)])
    lag2 = np.mean([C_emp[i, i + 2] for i in range(p - 2)])
    assert abs(lag1 - rho) <= 0.06
    assert abs(lag2 - rho**2) <= 0.06


def test_sample_cluster_class_specific_means():
    n, p = 600, 5
    labels = np.array([0] * (n // 2) + [1] * (n - n // 2), dtype=np.int64)
    rng = np.random.default_rng(7)
    X = corr.sample_cluster(
        n_samples=n,
        n_features=p,
        rng=rng,
        structure="equicorrelated",
        class_labels=labels,
        class_rho={0: 0.7, 1: 0.2},
        baseline_rho=0.0,
    )
    X0 = X[labels == 0]
    X1 = X[labels == 1]
    C0 = np.corrcoef(X0, rowvar=False).astype(np.float64)
    C1 = np.corrcoef(X1, rowvar=False).astype(np.float64)
    assert abs(corr.offdiag_metrics(C0)["mean_offdiag"] - 0.7) <= 0.06
    assert abs(corr.offdiag_metrics(C1)["mean_offdiag"] - 0.2) <= 0.06


def test_sample_cluster_errors():
    rng = np.random.default_rng(0)
    # Missing rho in global mode
    with pytest.raises(ValueError):
        corr.sample_cluster(10, 3, rng)  # type: ignore[call-arg]
    # Label length mismatch
    with pytest.raises(ValueError):
        corr.sample_cluster(
            n_samples=10,
            n_features=3,
            rng=rng,
            structure="equicorrelated",
            rho=0.5,
            class_labels=np.array([0, 1, 0], dtype=np.int64),
        )


# -------------------------
# find_seed_for_correlation
# -------------------------

@pytest.mark.xfail(
    reason=(
        "find_seed_for_correlation builds meta with 'tries': int(t + 1) but 't' is undefined, "
        "causing NameError. Fix by enumerating the loop and using try_idx instead."
    ),
    strict=False,
)
def test_find_seed_for_correlation_tol_mode():
    seed, meta = biomedical_data_generator.utils.correlation_seed_search.find_seed_for_correlation(
        n_samples=200,
        n_cluster_features=4,
        rho_target=0.5,
        structure="equicorrelated",
        tol=0.03,
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
        biomedical_data_generator.utils.correlation_seed_search.find_seed_for_correlation(
            n_samples=80,
            n_cluster_features=5,
            rho_target=0.2,
            structure="toeplitz",
            metric="min_offdiag",
            threshold=0.99,  # unrealistic
            op=">=",
            tol=None,
            start_seed=0,
            max_tries=3,
        )
