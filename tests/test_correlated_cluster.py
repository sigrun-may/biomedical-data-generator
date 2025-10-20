# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tests for correlated biomarker cluster generation."""

from typing import Literal

import numpy as np
import pytest

from biomedical_data_generator.config import CorrCluster
from biomedical_data_generator.features.correlated import (
    _cov_equicorr,
    _cov_toeplitz,
    find_seed_for_correlation,
    generate_correlated_cluster,
    sample_cluster_matrix,
)

Structure = Literal["equicorrelated", "toeplitz"]
STRUCTURES: tuple[Structure, ...] = ("equicorrelated", "toeplitz")


# ======================
# Helper functions
# ======================
def _mean_offdiag(corr: np.ndarray) -> float:
    """Compute mean of off-diagonal correlation elements."""
    if corr.ndim == 0:  # scalar (size=1 case)
        return 1.0
    if corr.shape[0] <= 1:
        return 1.0
    d = np.eye(corr.shape[0], dtype=bool)
    off = corr[~d]
    return float(off.mean()) if off.size else 1.0


def _min_offdiag(corr: np.ndarray) -> float:
    """Compute minimum of off-diagonal correlation elements."""
    if corr.ndim == 0:  # scalar (size=1 case)
        return 1.0
    if corr.shape[0] <= 1:
        return 1.0
    d = np.eye(corr.shape[0], dtype=bool)
    off = corr[~d]
    return float(off.min()) if off.size else 1.0


def _max_offdiag(corr: np.ndarray) -> float:
    """Compute maximum of off-diagonal correlation elements."""
    if corr.ndim == 0:  # scalar (size=1 case)
        return 1.0
    if corr.shape[0] <= 1:
        return 1.0
    d = np.eye(corr.shape[0], dtype=bool)
    off = corr[~d]
    return float(off.max()) if off.size else 1.0


# ======================
# Tests for covariance builders
# ======================
class TestCovarianceBuilders:
    """Test covariance matrix construction."""

    def test_cov_equicorr_shape(self):
        """Equicorrelated covariance has correct shape."""
        Sigma = _cov_equicorr(n_cluster_features=5, rho=0.7)
        assert Sigma.shape == (5, 5)
        assert Sigma.dtype == np.float64

    def test_cov_equicorr_diagonal(self):
        """Equicorrelated covariance has ones on diagonal."""
        Sigma = _cov_equicorr(n_cluster_features=4, rho=0.6)
        assert np.allclose(np.diag(Sigma), 1.0)

    def test_cov_equicorr_offdiagonal(self):
        """Equicorrelated covariance has rho off diagonal."""
        rho = 0.75
        Sigma = _cov_equicorr(n_cluster_features=6, rho=rho)
        # Check a few off-diagonal elements
        assert np.allclose(Sigma[0, 1], rho)
        assert np.allclose(Sigma[2, 4], rho)
        assert np.allclose(Sigma[1, 5], rho)

    def test_cov_equicorr_symmetric(self):
        """Equicorrelated covariance is symmetric."""
        Sigma = _cov_equicorr(n_cluster_features=5, rho=0.8)
        assert np.allclose(Sigma, Sigma.T)

    def test_cov_toeplitz_shape(self):
        """Toeplitz covariance has correct shape."""
        Sigma = _cov_toeplitz(n_cluster_features=4, rho=0.5)
        assert Sigma.shape == (4, 4)
        assert Sigma.dtype == np.float64

    def test_cov_toeplitz_diagonal(self):
        """Toeplitz covariance has ones on diagonal."""
        Sigma = _cov_toeplitz(n_cluster_features=5, rho=0.7)
        assert np.allclose(np.diag(Sigma), 1.0)

    def test_cov_toeplitz_decay(self):
        """Toeplitz covariance decays with distance."""
        rho = 0.8
        Sigma = _cov_toeplitz(n_cluster_features=6, rho=rho)
        # Distance 1: should be rho
        assert np.allclose(Sigma[0, 1], rho)
        # Distance 2: should be rho^2
        assert np.allclose(Sigma[0, 2], rho**2)
        # Distance 3: should be rho^3
        assert np.allclose(Sigma[0, 3], rho**3)

    def test_cov_toeplitz_symmetric(self):
        """Toeplitz covariance is symmetric."""
        Sigma = _cov_toeplitz(n_cluster_features=5, rho=0.6)
        assert np.allclose(Sigma, Sigma.T)

    def test_cov_toeplitz_negative_rho(self):
        """Toeplitz allows negative rho."""
        rho = -0.5
        Sigma = _cov_toeplitz(n_cluster_features=4, rho=rho)
        # Distance 1: negative
        assert Sigma[0, 1] < 0
        # Distance 2: positive (negative squared)
        assert Sigma[0, 2] > 0


# ======================
# Tests for sample_cluster_matrix
# ======================
class TestSampleClusterMatrix:
    """Test cluster matrix sampling."""

    def test_sample_cluster_matrix_shape(self):
        """Sampled matrix has correct shape."""
        cluster = CorrCluster(n_cluster_features=5, rho=0.7)
        rng = np.random.default_rng(42)
        X = sample_cluster_matrix(n=100, cluster=cluster, rng=rng)
        assert X.shape == (100, 5)

    def test_sample_cluster_matrix_standardized(self):
        """Columns are approximately standardized."""
        cluster = CorrCluster(n_cluster_features=4, rho=0.6)
        rng = np.random.default_rng(42)
        X = sample_cluster_matrix(n=500, cluster=cluster, rng=rng)

        # Means should be close to 0
        means = X.mean(axis=0)
        assert np.allclose(means, 0.0, atol=0.15)

        # Standard deviations should be close to 1
        stds = X.std(axis=0)
        assert np.allclose(stds, 1.0, atol=0.15)

    def test_sample_cluster_matrix_correlation_equicorr(self):
        """Equicorrelated structure achieves target correlation."""
        cluster = CorrCluster(n_cluster_features=6, rho=0.8, structure="equicorrelated")
        rng = np.random.default_rng(123)
        X = sample_cluster_matrix(n=1000, cluster=cluster, rng=rng)

        C = np.corrcoef(X, rowvar=False)
        mean_corr = _mean_offdiag(C)

        # Should be close to target (allow some sampling variation)
        assert 0.70 <= mean_corr <= 0.90

    def test_sample_cluster_matrix_correlation_toeplitz(self):
        """Toeplitz structure shows decaying correlation."""
        cluster = CorrCluster(n_cluster_features=5, rho=0.7, structure="toeplitz")
        rng = np.random.default_rng(456)
        X = sample_cluster_matrix(n=800, cluster=cluster, rng=rng)

        C = np.corrcoef(X, rowvar=False)

        # Adjacent pairs should have higher correlation than distant pairs
        adjacent = abs(C[0, 1])
        distant = abs(C[0, 4])
        assert adjacent > distant

    def test_sample_cluster_matrix_reproducibility(self):
        """Same seed produces identical results."""
        cluster = CorrCluster(n_cluster_features=4, rho=0.6)

        rng1 = np.random.default_rng(999)
        X1 = sample_cluster_matrix(n=100, cluster=cluster, rng=rng1)

        rng2 = np.random.default_rng(999)
        X2 = sample_cluster_matrix(n=100, cluster=cluster, rng=rng2)

        assert np.allclose(X1, X2)


# ======================
# Tests for generate_correlated_cluster
# ======================
class TestGenerateCorrelatedCluster:
    """Test standalone cluster generation."""

    @pytest.mark.parametrize("structure", STRUCTURES)
    def test_basic_generation(self, structure: Structure):
        """Basic cluster generation works for both structures."""
        X, meta = generate_correlated_cluster(n_samples=200, n_cluster_features=5, rho=0.6, structure=structure)

        assert X.shape == (200, 5)
        assert meta["n_cluster_features"] == 5
        assert meta["rho"] == 0.6
        assert meta["structure"] == structure

    def test_correlation_matrix_in_meta(self):
        """Metadata contains empirical correlation matrix."""
        X, meta = generate_correlated_cluster(n_samples=150, n_cluster_features=4, rho=0.7)

        assert "corr_matrix" in meta
        C = meta["corr_matrix"]
        assert isinstance(C, np.ndarray)
        assert C.shape == (4, 4)

        # Diagonal should be ones
        assert np.allclose(np.diag(C), 1.0)

    def test_mean_offdiag_calculation(self):
        """Mean off-diagonal correlation is calculated correctly."""
        X, meta = generate_correlated_cluster(n_samples=300, n_cluster_features=5, rho=0.75)

        assert "mean_offdiag" in meta
        mean_corr = meta["mean_offdiag"]

        # Should be close to target (allow sampling variation)
        assert 0.60 <= mean_corr <= 0.90

    def test_min_offdiag_calculation(self):
        """Minimum off-diagonal correlation is calculated."""
        X, meta = generate_correlated_cluster(n_samples=200, n_cluster_features=6, rho=0.8)

        assert "min_offdiag" in meta
        min_corr = meta["min_offdiag"]

        # Should be positive for positive rho
        assert min_corr > 0

    def test_label_in_meta(self):
        """Label is stored in metadata."""
        label = "Inflammation markers"
        X, meta = generate_correlated_cluster(n_samples=100, n_cluster_features=3, rho=0.6, label=label)

        assert meta["label"] == label

    def test_reproducibility_with_seed(self):
        """Same RNG seed produces identical clusters."""
        rng1 = np.random.default_rng(42)
        X1, meta1 = generate_correlated_cluster(n_samples=100, n_cluster_features=4, rho=0.7, rng=rng1)

        rng2 = np.random.default_rng(42)
        X2, meta2 = generate_correlated_cluster(n_samples=100, n_cluster_features=4, rho=0.7, rng=rng2)

        assert np.allclose(X1, X2)
        assert np.allclose(meta1["corr_matrix"], meta2["corr_matrix"])

    def test_size_one_cluster(self):
        """Size-1 cluster works (edge case)."""
        X, meta = generate_correlated_cluster(n_samples=50, n_cluster_features=1, rho=0.5)

        assert X.shape == (50, 1)
        assert meta["n_cluster_features"] == 1
        # Single feature: correlation matrix is scalar 1.0
        # mean_offdiag and min_offdiag should be 1.0 by convention
        assert meta["mean_offdiag"] == 1.0
        assert meta["min_offdiag"] == 1.0

    @pytest.mark.parametrize("rho", [0.3, 0.5, 0.7, 0.9])
    def test_varying_correlation_strengths(self, rho: float):
        """Different correlation strengths produce expected patterns."""
        X, meta = generate_correlated_cluster(n_samples=500, n_cluster_features=4, rho=rho)

        mean_corr = meta["mean_offdiag"]
        # Allow reasonable deviation due to sampling
        assert abs(mean_corr - rho) < 0.2

    def test_toeplitz_negative_rho(self):
        """Toeplitz structure works with negative rho."""
        X, meta = generate_correlated_cluster(n_samples=200, n_cluster_features=4, rho=-0.5, structure="toeplitz")

        C = meta["corr_matrix"]
        # Adjacent correlations should be negative
        assert C[0, 1] < 0
        # Distance-2 correlations should be positive
        assert C[0, 2] > 0


# ======================
# Tests for validation
# ======================
class TestValidation:
    """Test input validation and error handling."""

    def test_invalid_size_zero(self):
        """Size must be >= 1."""
        with pytest.raises(ValueError, match="n_cluster_features must be >= 1"):
            generate_correlated_cluster(n_samples=100, n_cluster_features=0, rho=0.5)

    def test_invalid_size_negative(self):
        """Negative size raises error."""
        with pytest.raises(ValueError, match="n_cluster_features must be >= 1"):
            generate_correlated_cluster(n_samples=100, n_cluster_features=-1, rho=0.5)

    def test_equicorr_rho_too_large(self):
        """Equicorrelated rho must be < 1."""
        with pytest.raises(ValueError, match="rho must be in"):
            generate_correlated_cluster(n_samples=100, n_cluster_features=3, rho=1.0, structure="equicorrelated")

    def test_equicorr_negative_rho(self):
        """Equicorrelated doesn't allow negative rho."""
        with pytest.raises(ValueError, match="rho must be in"):
            generate_correlated_cluster(n_samples=100, n_cluster_features=3, rho=-0.5, structure="equicorrelated")

    def test_toeplitz_rho_too_large(self):
        """Toeplitz rho must have |rho| < 1."""
        with pytest.raises(ValueError, match="must be < 1"):
            generate_correlated_cluster(n_samples=100, n_cluster_features=3, rho=1.0, structure="toeplitz")

    def test_toeplitz_rho_too_small(self):
        """Toeplitz rho must have |rho| < 1."""
        with pytest.raises(ValueError, match="must be < 1"):
            generate_correlated_cluster(n_samples=100, n_cluster_features=3, rho=-1.0, structure="toeplitz")


# ======================
# Tests for find_seed_for_correlation
# ======================
class TestFindSeedForCorrelation:
    """Test automatic seed search functionality."""

    def test_finds_seed_within_tolerance(self):
        """Find seed that achieves correlation within tolerance."""
        seed, meta = find_seed_for_correlation(
            n_samples=300, n_cluster_features=4, rho_target=0.7, tol=0.05, max_tries=100
        )

        assert isinstance(seed, int)
        assert seed >= 0

        mean_corr = meta["mean_offdiag"]
        assert abs(mean_corr - 0.7) <= 0.05

    def test_finds_seed_above_threshold(self):
        """Find seed where metric exceeds threshold."""
        seed, meta = find_seed_for_correlation(
            n_samples=400,
            n_cluster_features=5,
            rho_target=0.8,
            metric="mean_offdiag",
            threshold=0.75,
            tol=None,
            max_tries=200,
        )

        assert meta["mean_offdiag"] >= 0.75

    def test_min_offdiag_metric(self):
        """Can optimize minimum off-diagonal correlation."""
        seed, meta = find_seed_for_correlation(
            n_samples=300,
            n_cluster_features=4,
            rho_target=0.7,
            metric="min_offdiag",
            threshold=0.60,
            tol=None,
            max_tries=200,
        )

        assert meta["min_offdiag"] >= 0.60

    def test_tolerance_takes_precedence(self):
        """Tolerance criterion takes precedence over threshold."""
        seed, meta = find_seed_for_correlation(
            n_samples=200, n_cluster_features=3, rho_target=0.6, tol=0.03, threshold=0.50, max_tries=150
        )

        # Should satisfy tolerance
        assert abs(meta["mean_offdiag"] - 0.6) <= 0.03

    def test_reproducibility_from_start_seed(self):
        """Same start_seed produces same result."""
        seed1, meta1 = find_seed_for_correlation(
            n_samples=200, n_cluster_features=4, rho_target=0.7, tol=0.04, start_seed=100, max_tries=50
        )

        seed2, meta2 = find_seed_for_correlation(
            n_samples=200, n_cluster_features=4, rho_target=0.7, tol=0.04, start_seed=100, max_tries=50
        )

        assert seed1 == seed2

    def test_fails_with_impossible_criteria(self):
        """Raises error if criteria cannot be met."""
        with pytest.raises(RuntimeError, match="Failed to find seed"):
            find_seed_for_correlation(
                n_samples=30,  # small
                n_cluster_features=250,  # large cluster
                rho_target=0.98,  # very high
                tol=0.001,  # very tight
                max_tries=5,  # few tries
            )

    def test_works_for_toeplitz(self):
        """Seed search works for Toeplitz structure."""
        seed, meta = find_seed_for_correlation(
            n_samples=30, n_cluster_features=5, rho_target=0.6, structure="toeplitz", tol=0.05, max_tries=200
        )

        assert meta["structure"] == "toeplitz"
        assert abs(meta["mean_offdiag"] - 0.6) <= 0.05

    def test_invalid_rho_target(self):
        """Invalid rho_target raises error."""
        with pytest.raises(ValueError, match="rho_target must be in"):
            find_seed_for_correlation(n_samples=100, n_cluster_features=3, rho_target=1.5, max_tries=10)

    def test_invalid_size(self):
        """Invalid size raises error."""
        with pytest.raises(ValueError, match="n_cluster_features must be >= 1"):
            find_seed_for_correlation(n_samples=100, n_cluster_features=0, rho_target=0.6, max_tries=10)


# ======================
# Integration tests
# ======================
class TestIntegration:
    """Integration tests combining multiple components."""

    def test_cluster_in_dataset_context(self):
        """Cluster can be used in CorrCluster config."""
        cluster = CorrCluster(
            n_cluster_features=5,
            rho=0.75,
            anchor_role="informative",
            anchor_effect_size="large",
            label="Cytokine panel",
        )

        rng = np.random.default_rng(42)
        X = sample_cluster_matrix(n=200, cluster=cluster, rng=rng)

        assert X.shape == (200, 5)

        # Verify correlation
        C = np.corrcoef(X, rowvar=False)
        mean_corr = _mean_offdiag(C)
        assert 0.65 <= mean_corr <= 0.85

    def test_multiple_clusters_independent(self):
        """Multiple clusters with different seeds are independent."""
        cluster1 = CorrCluster(n_cluster_features=3, rho=0.7, random_state=1)
        cluster2 = CorrCluster(n_cluster_features=3, rho=0.7, random_state=2)

        rng1 = np.random.default_rng(cluster1.random_state)
        rng2 = np.random.default_rng(cluster2.random_state)

        X1 = sample_cluster_matrix(n=200, cluster=cluster1, rng=rng1)
        X2 = sample_cluster_matrix(n=200, cluster=cluster2, rng=rng2)

        # Clusters should be different
        assert not np.allclose(X1, X2)

        # Each cluster should have internal correlation
        C1 = np.corrcoef(X1, rowvar=False)
        C2 = np.corrcoef(X2, rowvar=False)

        assert _mean_offdiag(C1) > 0.5
        assert _mean_offdiag(C2) > 0.5

    def test_teaching_workflow(self):
        """Typical teaching workflow: generate, inspect, find better seed."""
        # Initial generation
        X1, meta1 = generate_correlated_cluster(n_samples=150, n_cluster_features=4, rho=0.8, label="Initial attempt")

        initial_corr = meta1["mean_offdiag"]

        # Find better seed if needed
        if abs(initial_corr - 0.8) > 0.05:
            seed, meta2 = find_seed_for_correlation(
                n_samples=150, n_cluster_features=4, rho_target=0.8, tol=0.03, max_tries=100
            )

            improved_corr = meta2["mean_offdiag"]
            assert abs(improved_corr - 0.8) <= 0.03


# ======================
# Performance tests
# ======================
class TestPerformance:
    """Test performance characteristics."""

    def test_large_sample_size(self):
        """Handles large sample sizes efficiently."""
        X, meta = generate_correlated_cluster(n_samples=10000, n_cluster_features=5, rho=0.7)

        assert X.shape == (10000, 5)

    def test_large_cluster_size(self):
        """Handles large cluster sizes."""
        X, meta = generate_correlated_cluster(n_samples=200, n_cluster_features=20, rho=0.6)

        assert X.shape == (200, 20)

    @pytest.mark.slow
    def test_seed_search_efficiency(self):
        """Seed search completes in reasonable time."""
        seed, meta = find_seed_for_correlation(
            n_samples=500, n_cluster_features=8, rho_target=0.75, tol=0.04, max_tries=300
        )

        # Should find solution
        assert isinstance(seed, int)


# ======================
# Edge case tests
# ======================
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_correlation(self):
        """Zero correlation produces independent features."""
        X, meta = generate_correlated_cluster(n_samples=500, n_cluster_features=4, rho=0.0)

        C = np.corrcoef(X, rowvar=False)
        # Off-diagonal should be close to zero
        np.fill_diagonal(C, 0)
        assert np.abs(C).max() < 0.2  # Allow some sampling variation

    def test_very_high_correlation(self):
        """Very high correlation (but < 1) works."""
        X, meta = generate_correlated_cluster(n_samples=300, n_cluster_features=3, rho=0.99)

        mean_corr = meta["mean_offdiag"]
        assert mean_corr > 0.95

    def test_small_sample_size(self):
        """Small sample sizes work but may have high variance."""
        X, meta = generate_correlated_cluster(n_samples=20, n_cluster_features=3, rho=0.7)

        assert X.shape == (20, 3)
        # Correlation may be less precise with small n
        assert 0.3 <= meta["mean_offdiag"] <= 1.0

    def test_very_large_cluster(self):
        """Very large clusters work."""
        X, meta = generate_correlated_cluster(n_samples=100, n_cluster_features=50, rho=0.5)

        assert X.shape == (100, 50)
        assert meta["corr_matrix"].shape == (50, 50)

    def test_toeplitz_weak_correlation(self):
        """Toeplitz with weak correlation."""
        X, meta = generate_correlated_cluster(n_samples=200, n_cluster_features=5, rho=0.2, structure="toeplitz")

        C = meta["corr_matrix"]
        # Distant features should have very weak correlation
        assert abs(C[0, 4]) < 0.2

    def test_numerical_stability_extreme_values(self):
        """Numerically stable with edge correlation values."""
        # Very close to 1
        X1, _ = generate_correlated_cluster(n_samples=100, n_cluster_features=3, rho=0.999)
        assert not np.any(np.isnan(X1))
        assert not np.any(np.isinf(X1))

        # Very close to 0
        X2, _ = generate_correlated_cluster(n_samples=100, n_cluster_features=3, rho=0.001)
        assert not np.any(np.isnan(X2))
        assert not np.any(np.isinf(X2))


class TestRNGHandling:
    """Test random number generator handling."""

    def test_none_rng_creates_new_generator(self):
        """Passing None as RNG creates new generator."""
        X1, _ = generate_correlated_cluster(n_samples=50, n_cluster_features=3, rho=0.6, rng=None)

        X2, _ = generate_correlated_cluster(n_samples=50, n_cluster_features=3, rho=0.6, rng=None)

        # Should be different (different random states)
        assert not np.allclose(X1, X2)

    def test_explicit_rng_reproducibility(self):
        """Explicit RNG with same state produces same result."""
        rng1 = np.random.default_rng(42)
        X1, _ = generate_correlated_cluster(n_samples=50, n_cluster_features=3, rho=0.6, rng=rng1)

        rng2 = np.random.default_rng(42)
        X2, _ = generate_correlated_cluster(n_samples=50, n_cluster_features=3, rho=0.6, rng=rng2)

        assert np.allclose(X1, X2)
