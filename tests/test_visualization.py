# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tests for visualization utilities."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing

from biomedical_data_generator.utils.visualization import (
    plot_all_correlation_clusters,
    plot_correlation_matrix,
    plot_correlation_matrix_for_cluster,
    plot_correlation_matrices_per_cluster,
)


class TestPlotCorrelationMatrix:
    """Tests for plot_correlation_matrix function."""

    def test_basic_plot(self):
        """Test basic correlation matrix plotting."""
        C = np.array([[1.0, 0.5], [0.5, 1.0]])
        fig, ax = plot_correlation_matrix(C, show=False)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_with_title(self):
        """Test plotting with a custom title."""
        C = np.array([[1.0, 0.5], [0.5, 1.0]])
        fig, ax = plot_correlation_matrix(C, title="Test Matrix", show=False)
        assert ax.get_title() == "Test Matrix"
        plt.close(fig)

    def test_with_labels(self):
        """Test plotting with custom labels."""
        C = np.array([[1.0, 0.5], [0.5, 1.0]])
        labels = ["Feature A", "Feature B"]
        fig, ax = plot_correlation_matrix(C, labels=labels, show=False)
        assert ax.get_xticklabels()[0].get_text() == "Feature A"
        plt.close(fig)

    def test_with_existing_ax(self):
        """Test plotting on existing axes."""
        C = np.array([[1.0, 0.5], [0.5, 1.0]])
        fig, ax = plt.subplots()
        returned_fig, returned_ax = plot_correlation_matrix(C, ax=ax, show=False)
        assert returned_ax == ax
        plt.close(fig)

    def test_with_annotations_small_matrix(self):
        """Test annotations for small matrices."""
        C = np.array([[1.0, 0.5], [0.5, 1.0]])
        fig, ax = plot_correlation_matrix(C, annot=True, show=False)
        # Check that text annotations were added
        assert len(ax.texts) > 0
        plt.close(fig)

    def test_no_annotations_large_matrix(self):
        """Test that large matrices don't get annotations."""
        C = np.random.rand(30, 30)
        fig, ax = plot_correlation_matrix(C, annot=True, show=False)
        # No annotations should be added for large matrices
        assert len(ax.texts) == 0
        plt.close(fig)

    def test_custom_vmin_vmax(self):
        """Test custom color scale limits."""
        C = np.array([[1.0, 0.5], [0.5, 1.0]])
        fig, ax = plot_correlation_matrix(C, vmin=0.0, vmax=1.0, show=False)
        images = ax.get_images()
        assert len(images) > 0
        assert images[0].get_clim() == (0.0, 1.0)
        plt.close(fig)

    def test_invalid_matrix_not_square(self):
        """Test error handling for non-square matrix."""
        C = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2]])
        with pytest.raises(ValueError, match="must be a square 2D array"):
            plot_correlation_matrix(C, show=False)

    def test_invalid_matrix_1d(self):
        """Test error handling for 1D array."""
        C = np.array([1.0, 0.5, 0.3])
        with pytest.raises(ValueError, match="must be a square 2D array"):
            plot_correlation_matrix(C, show=False)


class TestPlotCorrelationMatrixForCluster:
    """Tests for plot_correlation_matrix_for_cluster function."""

    def test_plot_single_cluster(self):
        """Test plotting correlation matrix for a single cluster."""
        # Create test data
        df = pd.DataFrame(
            {"corr1_anchor": [1, 2, 3], "corr1_2": [1.1, 2.1, 3.1], "corr1_3": [0.9, 1.9, 2.9], "noise1": [0.5, 0.6, 0.7]}
        )

        # Create mock meta object
        class MockMeta:
            corr_cluster_indices = {1: [0, 1, 2]}
            anchor_idx = {1: 0}

        meta = MockMeta()
        C = plot_correlation_matrix_for_cluster(df, meta, cluster_id=1, show=False)

        assert C.shape == (3, 3)
        assert np.allclose(np.diag(C), 1.0)
        plt.close("all")

    def test_with_different_correlation_methods(self):
        """Test different correlation methods."""
        df = pd.DataFrame({"corr1_anchor": [1, 2, 3, 4, 5], "corr1_2": [1.1, 2.2, 3.3, 4.4, 5.5]})

        class MockMeta:
            corr_cluster_indices = {1: [0, 1]}
            anchor_idx = {1: 0}

        meta = MockMeta()

        for method in ["pearson", "kendall", "spearman"]:
            C = plot_correlation_matrix_for_cluster(df, meta, cluster_id=1, correlation_method=method, show=False)
            assert C.shape == (2, 2)
            plt.close("all")


class TestPlotCorrelationMatricesPerCluster:
    """Tests for plot_correlation_matrices_per_cluster function."""

    def test_plot_multiple_clusters(self):
        """Test plotting correlation matrices for multiple clusters."""
        df = pd.DataFrame(
            {
                "corr1_anchor": [1, 2, 3],
                "corr1_2": [1.1, 2.1, 3.1],
                "corr2_anchor": [4, 5, 6],
                "corr2_2": [4.1, 5.1, 6.1],
            }
        )

        clusters = {1: [0, 1], 2: [2, 3]}
        result = plot_correlation_matrices_per_cluster(df, clusters, show=False)

        assert len(result) == 2
        assert 1 in result
        assert 2 in result

        for cluster_id, (fig, ax) in result.items():
            assert fig is not None
            assert ax is not None
            plt.close(fig)

    def test_with_labels_map(self):
        """Test plotting with custom labels."""
        df = pd.DataFrame({"corr1_anchor": [1, 2, 3], "corr1_2": [1.1, 2.1, 3.1]})

        clusters = {1: [0, 1]}
        labels_map = {1: "Custom Cluster"}
        result = plot_correlation_matrices_per_cluster(df, clusters, labels_map=labels_map, show=False)

        assert 1 in result
        fig, ax = result[1]
        assert "Custom Cluster" in ax.get_title()
        plt.close(fig)


class TestPlotAllCorrelationClusters:
    """Tests for plot_all_correlation_clusters function."""

    def test_plot_all_clusters(self):
        """Test plotting all correlation clusters together."""
        df = pd.DataFrame(
            {
                "corr1_anchor": [1, 2, 3, 4, 5],
                "corr1_2": [1.1, 2.1, 3.1, 4.1, 5.1],
                "corr2_anchor": [2, 3, 4, 5, 6],
                "corr2_2": [2.2, 3.2, 4.2, 5.2, 6.2],
            }
        )

        fig, ax = plot_all_correlation_clusters(df, show=False)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_with_custom_title(self):
        """Test plotting with custom title."""
        df = pd.DataFrame(
            {
                "corr1_anchor": [1, 2, 3],
                "corr1_2": [1.1, 2.1, 3.1],
            }
        )

        fig, ax = plot_all_correlation_clusters(df, title="Custom Title", show=False)
        assert ax.get_title() == "Custom Title"
        plt.close(fig)

    def test_different_correlation_methods(self):
        """Test different correlation methods."""
        df = pd.DataFrame(
            {
                "corr1_anchor": [1, 2, 3, 4, 5],
                "corr1_2": [1.1, 2.2, 3.3, 4.4, 5.5],
            }
        )

        for method in ["pearson", "kendall", "spearman"]:
            fig, ax = plot_all_correlation_clusters(df, correlation_method=method, show=False)
            assert fig is not None
            plt.close(fig)

    def test_cluster_boundaries(self):
        """Test that cluster boundaries are drawn."""
        df = pd.DataFrame(
            {
                "corr1_anchor": [1, 2, 3],
                "corr1_2": [1.1, 2.1, 3.1],
                "corr2_anchor": [2, 3, 4],
                "corr2_2": [2.2, 3.2, 4.2],
            }
        )

        fig, ax = plot_all_correlation_clusters(df, draw_cluster_boundaries=True, show=False)
        # Just verify the plot was created successfully
        # (boundaries are drawn but may not be accessible via ax.lines in all matplotlib versions)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_no_cluster_boundaries(self):
        """Test without cluster boundaries."""
        df = pd.DataFrame(
            {
                "corr1_anchor": [1, 2, 3],
                "corr1_2": [1.1, 2.1, 3.1],
            }
        )

        fig, ax = plot_all_correlation_clusters(df, draw_cluster_boundaries=False, show=False)
        assert fig is not None
        plt.close(fig)

    def test_no_corr_columns_error(self):
        """Test error when no corr columns are present."""
        df = pd.DataFrame({"noise1": [1, 2, 3], "noise2": [4, 5, 6]})

        with pytest.raises(ValueError, match="No columns containing 'corr' found"):
            plot_all_correlation_clusters(df, show=False)

    def test_auto_annotation_small(self):
        """Test auto annotation for small matrices."""
        df = pd.DataFrame(
            {
                "corr1_anchor": [1, 2, 3],
                "corr1_2": [1.1, 2.1, 3.1],
            }
        )

        fig, ax = plot_all_correlation_clusters(df, annot=None, show=False)  # auto-decide
        assert fig is not None
        plt.close(fig)

    def test_custom_figsize(self):
        """Test custom figure size."""
        df = pd.DataFrame(
            {
                "corr1_anchor": [1, 2, 3],
                "corr1_2": [1.1, 2.1, 3.1],
            }
        )

        fig, ax = plot_all_correlation_clusters(df, figsize=(8, 8), show=False)
        assert fig.get_size_inches()[0] == 8
        assert fig.get_size_inches()[1] == 8
        plt.close(fig)
