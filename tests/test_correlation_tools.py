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
