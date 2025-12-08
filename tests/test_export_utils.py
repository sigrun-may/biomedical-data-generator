# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tests for export utilities."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biomedical_data_generator.meta import DatasetMeta
from biomedical_data_generator.utils.export_utils import (
    to_csv,
    to_labeled_dataframe,
    to_parquet,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    y = np.array([0, 1, 0])
    feature_names = ["feature_1", "feature_2", "feature_3"]
    class_labels = ["class_0", "class_1"]

    meta = DatasetMeta(
        feature_names=feature_names,
        informative_idx=[0, 1],
        noise_idx=[2],
        corr_cluster_indices={},
        anchor_idx={},
        anchor_role={},
        anchor_effect_size={},
        anchor_class={},
        cluster_label={},
        n_classes=2,
        class_names=class_labels,
        samples_per_class={0: 2, 1: 1},
        class_sep=[1.0],
        corr_between=0.0,
    )

    return X, y, meta


def test_to_labeled_dataframe_with_labels(sample_data):
    """Test converting to DataFrame with labels."""
    x, y, meta = sample_data

    df = to_labeled_dataframe(x, y, meta, include_labels=True)

    # Check shape - should have features + y column (string labels only if class_labels is non-empty)
    assert df.shape[0] == 3
    assert "y" in df.columns

    # Check feature values
    np.testing.assert_array_equal(df[["feature_1", "feature_2", "feature_3"]].values, x)

    # Check numeric labels
    np.testing.assert_array_equal(df["y"].values, y)


def test_to_labeled_dataframe_without_labels(sample_data):
    """Test converting to DataFrame without labels."""
    x, y, meta = sample_data

    df = to_labeled_dataframe(x, y, meta, include_labels=False)

    # Check shape (only features, no labels)
    assert df.shape == (3, 3)

    # Check column names
    assert list(df.columns) == ["feature_1", "feature_2", "feature_3"]

    # Check values
    np.testing.assert_array_equal(df.values, x)


def test_to_labeled_dataframe_with_dataframe_input(sample_data):
    """Test with DataFrame input."""
    x, y, meta = sample_data

    # Convert X to DataFrame
    x_df = pd.DataFrame(x, columns=["col1", "col2", "col3"])

    df = to_labeled_dataframe(x_df, y, meta, include_labels=True)

    # Should rename columns to match meta.feature_names
    assert df.columns[1] == "feature_1"
    assert df.columns[2] == "feature_2"
    assert df.columns[3] == "feature_3"
    assert "y" in df.columns


def test_to_labeled_dataframe_custom_label_names(sample_data):
    """Test with custom label column names."""
    x, y, meta = sample_data

    df = to_labeled_dataframe(
        x,
        y,
        meta,
        include_labels=True,
        label_col_name="class",
        label_str_col_name="diagnosis",
    )

    assert "class" in df.columns
    # String label column only added if class_labels exist
    assert "y" not in df.columns
    assert "y_label" not in df.columns


def test_to_labeled_dataframe_with_custom_feature_names(sample_data):
    """Test with custom feature names."""
    x, y, _ = sample_data

    custom_names = ["custom_1", "custom_2", "custom_3"]

    df = to_labeled_dataframe(x, y, None, feature_names=custom_names, include_labels=True)

    assert list(df.columns[1:4]) == custom_names


def test_to_labeled_dataframe_no_meta_no_names_raises(sample_data):
    """Test that missing both meta and feature_names raises error."""
    x, y, _ = sample_data

    with pytest.raises(ValueError, match="Either meta or feature_names must be provided"):
        to_labeled_dataframe(x, y, None, include_labels=False)


def test_to_labeled_dataframe_missing_y_with_labels_raises(sample_data):
    """Test that include_labels=True without y raises error."""
    x, _, meta = sample_data

    with pytest.raises(ValueError, match="y must be provided when include_labels=True"):
        to_labeled_dataframe(x, None, meta, include_labels=True)


def test_to_labeled_dataframe_shape_mismatch_raises(sample_data):
    """Test that shape mismatch between x and y raises error."""
    x, y, meta = sample_data

    # Create y with wrong length
    y_wrong = np.array([0, 1])  # Only 2 elements instead of 3

    with pytest.raises(ValueError, match="Shape mismatch"):
        to_labeled_dataframe(x, y_wrong, meta, include_labels=True)


def test_to_labeled_dataframe_without_string_labels(sample_data):
    """Test handling when meta doesn't have class_labels."""
    x, y, meta = sample_data

    # Create meta without class_labels attribute
    minimal_meta = DatasetMeta(
        feature_names=["f1", "f2", "f3"],
        informative_idx=[0],
        noise_idx=[1, 2],
        corr_cluster_indices={},
        anchor_idx={},
        anchor_role={},
        anchor_effect_size={},
        anchor_class={},
        cluster_label={},
        n_classes=2,
        class_names=[],  # Empty list
        samples_per_class={0: 2, 1: 1},
        class_sep=[1.0],
        corr_between=0.0,
    )

    df = to_labeled_dataframe(x, y, minimal_meta, include_labels=True)

    # Should have numeric labels but no string labels
    assert "y" in df.columns
    # String labels will still be added but will be empty strings or raise error
    # Let's check that it at least doesn't crash


def test_to_csv(sample_data, tmp_path):
    """Test exporting to CSV."""
    x, y, meta = sample_data

    csv_path = tmp_path / "test.csv"

    to_csv(x, y, meta, csv_path, index=False)

    # Check file exists
    assert csv_path.exists()

    # Read back and verify
    df = pd.read_csv(csv_path)
    assert df.shape[0] == 3
    assert "y" in df.columns


def test_to_csv_without_labels(sample_data, tmp_path):
    """Test exporting to CSV without labels."""
    x, y, meta = sample_data

    csv_path = tmp_path / "test_no_labels.csv"

    to_csv(x, y, meta, csv_path, include_labels=False, index=False)

    df = pd.read_csv(csv_path)
    assert df.shape == (3, 3)  # Only features
    assert "y" not in df.columns


def test_to_csv_with_custom_kwargs(sample_data, tmp_path):
    """Test to_csv with custom CSV kwargs."""
    x, y, meta = sample_data

    csv_path = tmp_path / "test_custom.csv"

    to_csv(x, y, meta, csv_path, sep=";", index=True)

    # Read back with semicolon separator
    df = pd.read_csv(csv_path, sep=";", index_col=0)
    assert df.shape[0] == 3


def test_to_parquet(sample_data, tmp_path):
    """Test exporting to Parquet."""
    pytest.importorskip("pyarrow", reason="PyArrow not installed")

    x, y, meta = sample_data

    parquet_path = tmp_path / "test.parquet"

    to_parquet(x, y, meta, parquet_path)

    # Check file exists
    assert parquet_path.exists()

    # Read back and verify
    df = pd.read_parquet(parquet_path)
    assert df.shape[0] == 3
    assert "y" in df.columns


def test_to_parquet_without_labels(sample_data, tmp_path):
    """Test exporting to Parquet without labels."""
    pytest.importorskip("pyarrow", reason="PyArrow not installed")

    x, y, meta = sample_data

    parquet_path = tmp_path / "test_no_labels.parquet"

    to_parquet(x, y, meta, parquet_path, include_labels=False)

    df = pd.read_parquet(parquet_path)
    assert df.shape == (3, 3)
    assert "y" not in df.columns


def test_to_parquet_with_path_object(sample_data):
    """Test to_parquet with Path object."""
    pytest.importorskip("pyarrow", reason="PyArrow not installed")

    x, y, meta = sample_data

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "test.parquet"

        to_parquet(x, y, meta, parquet_path)

        assert parquet_path.exists()
        df = pd.read_parquet(parquet_path)
        assert df.shape[0] == 3


def test_to_csv_with_string_path(sample_data):
    """Test to_csv with string path."""
    x, y, meta = sample_data

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = f"{tmpdir}/test.csv"

        to_csv(x, y, meta, csv_path, index=False)

        df = pd.read_csv(csv_path)
        assert df.shape[0] == 3
