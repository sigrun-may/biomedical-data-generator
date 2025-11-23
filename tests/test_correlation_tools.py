# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tests for correlation_tools module."""

import numpy as np
import pandas as pd
import pytest

from biomedical_data_generator.utils.correlation_tools import (
    compute_correlation_matrix,
    compute_correlation_metrics,
    get_cluster_frame,
)


class TestComputeCorrelationMetrics:
    """Tests for compute_correlation_metrics function."""

    def test_with_valid_matrix(self):
        """Test compute_correlation_metrics with a valid correlation matrix."""
        C = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
        metrics = compute_correlation_metrics(C)

        assert "mean_offdiag" in metrics
        assert "std_offdiag" in metrics
        assert "min_offdiag" in metrics
        assert "max_offdiag" in metrics
        assert "range_offdiag" in metrics
        assert "n_offdiag" in metrics

        # Verify values make sense
        assert metrics["n_offdiag"] == 6  # 3x3 matrix has 6 off-diagonal elements
        assert 0.3 <= metrics["mean_offdiag"] <= 0.5
        assert metrics["min_offdiag"] == 0.3
        assert metrics["max_offdiag"] == 0.5

    def test_with_single_feature(self):
        """Test compute_correlation_metrics with 1x1 matrix."""
        C = np.array([[1.0]])
        metrics = compute_correlation_metrics(C)

        assert metrics["mean_offdiag"] == 1.0
        assert metrics["std_offdiag"] == 0.0
        assert metrics["min_offdiag"] == 1.0
        assert metrics["max_offdiag"] == 1.0
        assert metrics["range_offdiag"] == 0.0
        assert metrics["n_offdiag"] == 0

    def test_error_non_square_matrix(self):
        """Test error when matrix is not square."""
        C = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4]])
        with pytest.raises(ValueError, match="corr_matrix must be a square 2D array"):
            compute_correlation_metrics(C)


class TestComputeCorrelationMatrix:
    """Tests for compute_correlation_matrix function."""

    def test_with_dataframe_pearson(self):
        """Test compute_correlation_matrix with DataFrame using pearson method."""
        df = pd.DataFrame(
            {
                "feat1": [1, 2, 3, 4, 5],
                "feat2": [2, 4, 6, 8, 10],
                "feat3": [1, 3, 2, 5, 4],
            }
        )
        C, labels = compute_correlation_matrix(df, method="pearson")

        assert C.shape == (3, 3)
        assert len(labels) == 3
        assert labels == ["feat1", "feat2", "feat3"]
        assert np.allclose(np.diag(C), 1.0)

    def test_with_dataframe_spearman(self):
        """Test compute_correlation_matrix with DataFrame using spearman method."""
        df = pd.DataFrame(
            {
                "feat1": [1, 2, 3, 4, 5],
                "feat2": [2, 4, 6, 8, 10],
            }
        )
        C, labels = compute_correlation_matrix(df, method="spearman")

        assert C.shape == (2, 2)
        assert len(labels) == 2
        assert np.allclose(np.diag(C), 1.0)

    def test_with_numpy_array_as_dataframe(self):
        """Test compute_correlation_matrix with numpy array converted to DataFrame."""
        X = np.random.randn(50, 4)
        df = pd.DataFrame(X, columns=["f0", "f1", "f2", "f3"])
        C, labels = compute_correlation_matrix(df, method="pearson")

        assert C.shape == (4, 4)
        assert len(labels) == 4
        assert labels == ["f0", "f1", "f2", "f3"]
        assert np.allclose(np.diag(C), 1.0)


class TestGetClusterFrame:
    """Tests for get_cluster_frame function."""

    def test_get_cluster_frame_basic(self):
        """Test get_cluster_frame with basic cluster."""
        df = pd.DataFrame(
            {
                "corr1_anchor": [1, 2, 3],
                "corr1_2": [1.1, 2.1, 3.1],
                "corr1_3": [0.9, 1.9, 2.9],
                "noise1": [0.5, 0.6, 0.7],
            }
        )

        class MockMeta:
            corr_cluster_indices = {1: [0, 1, 2]}
            anchor_idx = {1: 0}

        meta = MockMeta()
        result = get_cluster_frame(df, meta, cluster_id=1)

        assert len(result.columns) == 3
        assert "corr1_anchor" in result.columns
        assert "corr1_2" in result.columns
        assert "corr1_3" in result.columns

    def test_get_cluster_frame_anchor_first(self):
        """Test get_cluster_frame with anchor_first=True."""
        df = pd.DataFrame(
            {
                "corr1_2": [1.1, 2.1, 3.1],
                "corr1_anchor": [1, 2, 3],
                "corr1_3": [0.9, 1.9, 2.9],
            }
        )

        class MockMeta:
            corr_cluster_indices = {1: [0, 1, 2]}
            anchor_idx = {1: 1}

        meta = MockMeta()
        result = get_cluster_frame(df, meta, cluster_id=1, anchor_first=True)

        # Check that anchor column is first
        assert result.columns[0] == "corr1_anchor"

    def test_get_cluster_frame_no_anchor(self):
        """Test get_cluster_frame when cluster has no anchor."""
        df = pd.DataFrame(
            {
                "corr1_2": [1.1, 2.1, 3.1],
                "corr1_3": [0.9, 1.9, 2.9],
                "corr1_4": [1.2, 2.2, 3.2],
            }
        )

        class MockMeta:
            corr_cluster_indices = {1: [0, 1, 2]}
            anchor_idx = {1: None}

        meta = MockMeta()
        result = get_cluster_frame(df, meta, cluster_id=1, anchor_first=True)

        assert len(result.columns) == 3


class TestValidateCorrelation:
    """Tests for correlation validation in find_seed_for_correlation."""

    def test_equicorrelated_p1_valid(self):
        """Test validation for equicorrelated with p=1."""
        from biomedical_data_generator.utils.correlation_tools import _validate_correlation

        # Should not raise for valid correlation
        _validate_correlation("equicorrelated", p=1, correlation=0.5)
        _validate_correlation("equicorrelated", p=1, correlation=-0.5)

    def test_equicorrelated_p1_invalid(self):
        """Test validation for equicorrelated with p=1 and invalid correlation."""
        from biomedical_data_generator.utils.correlation_tools import _validate_correlation

        # Should raise for correlation >= 1.0 or <= -1.0
        with pytest.raises(ValueError, match="For p=1 require"):
            _validate_correlation("equicorrelated", p=1, correlation=1.0)
        with pytest.raises(ValueError, match="For p=1 require"):
            _validate_correlation("equicorrelated", p=1, correlation=-1.0)

    def test_equicorrelated_p2_valid(self):
        """Test validation for equicorrelated with p>=2."""
        from biomedical_data_generator.utils.correlation_tools import _validate_correlation

        # Should not raise for valid correlation
        _validate_correlation("equicorrelated", p=3, correlation=0.8)
        _validate_correlation("equicorrelated", p=3, correlation=-0.4)

    def test_equicorrelated_p2_invalid_lower_bound(self):
        """Test validation for equicorrelated with correlation below lower bound."""
        from biomedical_data_generator.utils.correlation_tools import _validate_correlation

        # For p=3, lower bound is -1/(3-1) = -0.5
        with pytest.raises(ValueError, match="Equicorrelated requires"):
            _validate_correlation("equicorrelated", p=3, correlation=-0.6)

    def test_equicorrelated_p2_invalid_upper_bound(self):
        """Test validation for equicorrelated with correlation >= 1.0."""
        from biomedical_data_generator.utils.correlation_tools import _validate_correlation

        with pytest.raises(ValueError, match="Equicorrelated requires"):
            _validate_correlation("equicorrelated", p=3, correlation=1.0)

    def test_toeplitz_valid(self):
        """Test validation for toeplitz structure."""
        from biomedical_data_generator.utils.correlation_tools import _validate_correlation

        # Should not raise for valid correlation
        _validate_correlation("toeplitz", p=5, correlation=0.7)
        _validate_correlation("toeplitz", p=5, correlation=-0.7)

    def test_toeplitz_invalid(self):
        """Test validation for toeplitz with invalid correlation."""
        from biomedical_data_generator.utils.correlation_tools import _validate_correlation

        # Should raise for |correlation| >= 1.0
        with pytest.raises(ValueError, match="Toeplitz requires"):
            _validate_correlation("toeplitz", p=5, correlation=1.0)
        with pytest.raises(ValueError, match="Toeplitz requires"):
            _validate_correlation("toeplitz", p=5, correlation=-1.0)


class TestPC1Share:
    """Tests for PC1 share functionality."""

    def test_pc1_share_with_dataframe(self):
        """Test pc1_share with DataFrame input."""
        from biomedical_data_generator.utils.correlation_tools import pc1_share

        df = pd.DataFrame(
            {
                "feat1": [1, 2, 3, 4, 5],
                "feat2": [2, 4, 6, 8, 10],
                "feat3": [1, 3, 2, 5, 4],
            }
        )
        evr = pc1_share(df, method="pearson")
        assert 0.0 <= evr <= 1.0
        assert isinstance(evr, float)

    def test_pc1_share_with_array(self):
        """Test pc1_share with numpy array."""
        from biomedical_data_generator.utils.correlation_tools import pc1_share

        X = np.random.randn(50, 5)
        evr = pc1_share(X, method="pearson")
        assert 0.0 <= evr <= 1.0

    def test_variance_partition_pc1(self):
        """Test variance_partition_pc1 function."""
        from biomedical_data_generator.utils.correlation_tools import variance_partition_pc1

        X = pd.DataFrame(np.random.randn(100, 5))
        metrics = variance_partition_pc1(X, method="pearson")

        assert "n_features" in metrics
        assert "pc1_evr" in metrics
        assert "pc1_var_ratio" in metrics
        assert metrics["n_features"] == 5
        assert 0.0 <= metrics["pc1_evr"] <= 1.0
