# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tests for correlation matrix construction."""

import numpy as np
import pytest

from biomedical_data_generator.features.correlated import build_correlation_matrix


class TestBuildCorrelationMatrix:
    """Test correlation matrix construction for both equicorrelated and toeplitz structures."""

    # ==================
    # Basic structure tests
    # ==================

    def test_equicorr_shape_and_dtype(self):
        """Equicorrelated matrix has correct shape and dtype."""
        R = build_correlation_matrix(n_features=5, correlation=0.7, structure="equicorrelated")
        assert R.shape == (5, 5)
        assert R.dtype == np.float64

    def test_toeplitz_shape_and_dtype(self):
        """Toeplitz matrix has correct shape and dtype."""
        R = build_correlation_matrix(n_features=4, correlation=0.5, structure="toeplitz")
        assert R.shape == (4, 4)
        assert R.dtype == np.float64

    # ==================
    # Diagonal tests
    # ==================

    def test_equicorr_diagonal_is_one(self):
        """Equicorrelated matrix has ones on diagonal."""
        R = build_correlation_matrix(n_features=6, correlation=0.6, structure="equicorrelated")
        assert np.allclose(np.diag(R), 1.0)

    def test_toeplitz_diagonal_is_one(self):
        """Toeplitz matrix has ones on diagonal."""
        R = build_correlation_matrix(n_features=5, correlation=0.7, structure="toeplitz")
        assert np.allclose(np.diag(R), 1.0)

    # ==================
    # Off-diagonal pattern tests
    # ==================

    def test_equicorr_offdiagonal_constant(self):
        """Equicorrelated matrix has constant off-diagonal values equal to correlation."""
        correlation = 0.75
        R = build_correlation_matrix(n_features=6, correlation=correlation, structure="equicorrelated")

        # All off-diagonal elements should equal correlation
        mask = ~np.eye(6, dtype=bool)
        assert np.allclose(R[mask], correlation)

        # Check specific pairs
        assert np.allclose(R[0, 1], correlation)
        assert np.allclose(R[2, 4], correlation)
        assert np.allclose(R[1, 5], correlation)

    def test_toeplitz_exponential_decay(self):
        """Toeplitz matrix shows exponential decay with distance."""
        correlation = 0.8
        R = build_correlation_matrix(n_features=6, correlation=correlation, structure="toeplitz")

        # Distance 1: should be correlation
        assert np.allclose(R[0, 1], correlation)
        # Distance 2: should be correlation^2
        assert np.allclose(R[0, 2], correlation**2)
        # Distance 3: should be correlation^3
        assert np.allclose(R[0, 3], correlation**3)
        # Distance 4: should be correlation^4
        assert np.allclose(R[0, 4], correlation**4)
        # Distance 5: should be correlation^5
        assert np.allclose(R[0, 5], correlation**5)

    def test_toeplitz_negative_correlation_alternating_signs(self):
        """Toeplitz with negative correlation produces alternating signs."""
        correlation = -0.5
        R = build_correlation_matrix(n_features=5, correlation=correlation, structure="toeplitz")

        # Distance 1: negative
        assert R[0, 1] < 0
        assert np.allclose(R[0, 1], correlation)

        # Distance 2: positive (negative squared)
        assert R[0, 2] > 0
        assert np.allclose(R[0, 2], correlation**2)

        # Distance 3: negative (negative cubed)
        assert R[0, 3] < 0
        assert np.allclose(R[0, 3], correlation**3)

    # ==================
    # Symmetry tests
    # ==================

    def test_equicorr_symmetric(self):
        """Equicorrelated matrix is symmetric."""
        R = build_correlation_matrix(n_features=5, correlation=0.8, structure="equicorrelated")
        assert np.allclose(R, R.T)

    def test_toeplitz_symmetric(self):
        """Toeplitz matrix is symmetric."""
        R = build_correlation_matrix(n_features=6, correlation=0.6, structure="toeplitz")
        assert np.allclose(R, R.T)

    # ==================
    # Edge case tests
    # ==================

    def test_single_feature_returns_scalar_one(self):
        """Single feature (n=1) returns matrix [[1.0]]."""
        R_eq = build_correlation_matrix(n_features=1, correlation=0.5, structure="equicorrelated")
        R_tp = build_correlation_matrix(n_features=1, correlation=0.5, structure="toeplitz")

        assert R_eq.shape == (1, 1)
        assert R_tp.shape == (1, 1)
        assert np.allclose(R_eq, [[1.0]])
        assert np.allclose(R_tp, [[1.0]])

    def test_zero_correlation_equicorr(self):
        """Equicorrelated with correlation=0 produces identity matrix."""
        R = build_correlation_matrix(n_features=4, correlation=0.0, structure="equicorrelated")
        assert np.allclose(R, np.eye(4))

    def test_zero_correlation_toeplitz(self):
        """Toeplitz with correlation=0 produces identity matrix."""
        R = build_correlation_matrix(n_features=4, correlation=0.0, structure="toeplitz")
        assert np.allclose(R, np.eye(4))

    def test_very_high_correlation_equicorr(self):
        """Equicorrelated with correlation close to 1 is numerically stable."""
        R = build_correlation_matrix(n_features=5, correlation=0.999, structure="equicorrelated")

        # Should not have NaN or Inf
        assert not np.any(np.isnan(R))
        assert not np.any(np.isinf(R))

        # Diagonal should still be 1
        assert np.allclose(np.diag(R), 1.0)

        # Off-diagonal should be close to 0.999
        mask = ~np.eye(5, dtype=bool)
        assert np.allclose(R[mask], 0.999)

    def test_very_high_correlation_toeplitz(self):
        """Toeplitz with correlation close to 1 is numerically stable."""
        R = build_correlation_matrix(n_features=5, correlation=0.999, structure="toeplitz")

        # Should not have NaN or Inf
        assert not np.any(np.isnan(R))
        assert not np.any(np.isinf(R))

        # Diagonal should still be 1
        assert np.allclose(np.diag(R), 1.0)

    def test_very_low_correlation_both_structures(self):
        """Very low correlation (close to 0) is numerically stable."""
        for structure in ["equicorrelated", "toeplitz"]:
            R = build_correlation_matrix(n_features=4, correlation=0.001, structure=structure)

            # Should not have NaN or Inf
            assert not np.any(np.isnan(R))
            assert not np.any(np.isinf(R))

            # Should be close to identity
            assert np.allclose(R, np.eye(4), atol=0.01)

    def test_large_matrix_performance(self):
        """Large matrices (50x50) can be constructed efficiently."""
        R_eq = build_correlation_matrix(n_features=50, correlation=0.5, structure="equicorrelated")
        R_tp = build_correlation_matrix(n_features=50, correlation=0.5, structure="toeplitz")

        assert R_eq.shape == (50, 50)
        assert R_tp.shape == (50, 50)

        # Basic sanity checks
        assert np.allclose(np.diag(R_eq), 1.0)
        assert np.allclose(np.diag(R_tp), 1.0)

    # ==================
    # Validation tests (error cases)
    # ==================

    def test_invalid_n_features_zero_raises(self):
        """Zero features raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be positive"):
            build_correlation_matrix(n_features=0, correlation=0.5, structure="equicorrelated")

    def test_invalid_n_features_negative_raises(self):
        """Negative features raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be positive"):
            build_correlation_matrix(n_features=-5, correlation=0.5, structure="toeplitz")

    def test_equicorr_correlation_equals_one_raises(self):
        """Equicorrelated with correlation=1 raises ValueError (not PD)."""
        with pytest.raises(ValueError, match="Invalid correlation.*require.*< 1"):
            build_correlation_matrix(n_features=3, correlation=1.0, structure="equicorrelated")

    def test_equicorr_correlation_too_negative_raises(self):
        """Equicorrelated with correlation <= -1/(n-1) raises ValueError."""
        # For n=4, lower bound is -1/3 ≈ -0.333
        with pytest.raises(ValueError, match="Invalid correlation.*require.*< 1"):
            build_correlation_matrix(n_features=4, correlation=-0.5, structure="equicorrelated")

    def test_toeplitz_correlation_equals_one_raises(self):
        """Toeplitz with |correlation|=1 raises ValueError (not PD)."""
        with pytest.raises(ValueError, match="Invalid correlation.*require.*< 1"):
            build_correlation_matrix(n_features=5, correlation=1.0, structure="toeplitz")

        with pytest.raises(ValueError, match="Invalid correlation.*require.*< 1"):
            build_correlation_matrix(n_features=5, correlation=-1.0, structure="toeplitz")

    def test_toeplitz_correlation_exceeds_one_raises(self):
        """Toeplitz with |correlation| > 1 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid correlation.*require.*< 1"):
            build_correlation_matrix(n_features=3, correlation=1.5, structure="toeplitz")

        with pytest.raises(ValueError, match="Invalid correlation.*require.*< 1"):
            build_correlation_matrix(n_features=3, correlation=-1.5, structure="toeplitz")

    def test_unknown_structure_raises(self):
        """Unknown structure type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown structure"):
            build_correlation_matrix(n_features=3, correlation=0.5, structure="invalid")

    # ==================
    # Mathematical property tests
    # ==================

    def test_positive_definite_equicorr(self):
        """Equicorrelated matrix is positive definite for valid correlation."""
        R = build_correlation_matrix(n_features=5, correlation=0.7, structure="equicorrelated")

        # Check eigenvalues are all positive
        eigenvalues = np.linalg.eigvalsh(R)
        assert np.all(eigenvalues > 0), f"Matrix not positive definite: eigenvalues={eigenvalues}"

    def test_positive_definite_toeplitz(self):
        """Toeplitz matrix is positive definite for |correlation| < 1."""
        for correlation in [0.3, 0.7, -0.3, -0.7]:
            R = build_correlation_matrix(n_features=5, correlation=correlation, structure="toeplitz")

            # Check eigenvalues are all positive
            eigenvalues = np.linalg.eigvalsh(R)
            assert np.all(
                eigenvalues > 0
            ), f"Matrix not positive definite for correlation={correlation}: eigenvalues={eigenvalues}"

    def test_determinant_positive_both_structures(self):
        """Determinant is positive for valid correlation matrices."""
        for structure in ["equicorrelated", "toeplitz"]:
            R = build_correlation_matrix(n_features=4, correlation=0.6, structure=structure)
            det = np.linalg.det(R)
            assert det > 0, f"Determinant not positive for {structure}: det={det}"

    # ==================
    # Consistency tests
    # ==================

    def test_reproducibility(self):
        """Same parameters produce identical matrices."""
        R1 = build_correlation_matrix(n_features=5, correlation=0.65, structure="equicorrelated")
        R2 = build_correlation_matrix(n_features=5, correlation=0.65, structure="equicorrelated")
        assert np.allclose(R1, R2)

    def test_equicorr_independent_of_feature_order(self):
        """Equicorrelated matrix values don't depend on feature indices."""
        R = build_correlation_matrix(n_features=6, correlation=0.8, structure="equicorrelated")

        # All off-diagonal pairs should have same value
        mask = ~np.eye(6, dtype=bool)
        off_diag_values = R[mask]
        assert np.allclose(off_diag_values, 0.8)

    def test_toeplitz_depends_only_on_distance(self):
        """Toeplitz correlation depends only on distance, not absolute position."""
        R = build_correlation_matrix(n_features=8, correlation=0.7, structure="toeplitz")

        # Distance 2 should be same everywhere
        dist2_values = [R[i, i + 2] for i in range(6)]
        assert np.allclose(dist2_values, 0.7**2)

        # Distance 3 should be same everywhere
        dist3_values = [R[i, i + 3] for i in range(5)]
        assert np.allclose(dist3_values, 0.7**3)
