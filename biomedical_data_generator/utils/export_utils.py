# biomedical_data_generator/utils/export_utils.py

"""Export utilities for saving generated datasets to various formats."""

from __future__ import annotations

import pandas as pd
import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from biomedical_data_generator.meta import DatasetMeta

__all__ = [
    "to_labeled_dataframe",
    "to_csv",
    "to_parquet",
]


def to_labeled_dataframe(
        X: pd.DataFrame | NDArray[np.float64],
        y: NDArray[np.int64] | None = None,
        meta: DatasetMeta | None = None,
        *,
        include_labels: bool = True,
        label_col_name: str = "y",
        label_str_col_name: str = "y_label",
        feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Convert generated dataset to DataFrame with optional labels.

    Flexible conversion supporting multiple use cases:
    1. Full conversion: X + y + meta → df with features + labels
    2. Features only: X + meta → df with features (no labels)
    3. Custom names: override default column names

    Args:
        X: Feature matrix (DataFrame or ndarray).
        y: Optional class labels (integers 0 to n_classes-1).
        meta: Optional dataset metadata.
        include_labels: If True and y provided, add label columns.
        label_col_name: Column name for numeric labels.
        label_str_col_name: Column name for string labels.
        feature_names: Override meta.feature_names (for custom naming).

    Returns:
        DataFrame with requested columns.

    Raises:
        ValueError: If shapes mismatch or required args missing.

    Examples:
        >>> # Standard usage
        >>> df = to_labeled_dataframe(X, y, meta)

        >>> # Features only
        >>> df_features = to_labeled_dataframe(X, meta=meta, include_labels=False)

        >>> # Custom column names
        >>> df = to_labeled_dataframe(X, y, meta, 
        ...                   label_col_name="class",
        ...                   label_str_col_name="diagnosis")
    """
    # Determine feature names
    if feature_names is None:
        if meta is None:
            raise ValueError("Either meta or feature_names must be provided")
        feature_names = meta.feature_names

    # Convert X to DataFrame
    if isinstance(X, pd.DataFrame):
        df = X.copy()
        # Rename columns if needed
        if list(df.columns) != feature_names:
            df.columns = feature_names
    else:
        df = pd.DataFrame(X, columns=feature_names)

    # Add labels if requested
    if include_labels:
        if y is None:
            raise ValueError("y must be provided when include_labels=True")

        # Validate shape
        if df.shape[0] != len(y):
            raise ValueError(
                f"Shape mismatch: X has {df.shape[0]} samples but y has {len(y)}"
            )

        # Add numeric labels
        df[label_col_name] = y

        # Add string labels if meta available
        if meta is not None and hasattr(meta, "class_labels"):
            df[label_str_col_name] = [meta.class_labels[int(i)] for i in y]

    return df



def to_csv(
        X: pd.DataFrame | NDArray[np.float64],
        y: NDArray[np.int64],
        meta: DatasetMeta,
        filepath: str | Path,
        *,
        include_labels: bool = True,
        **csv_kwargs,
) -> None:
    """Export dataset to CSV file.

    Convenience wrapper around to_dataframe() + DataFrame.to_csv().

    Args:
        X: Feature matrix.
        y: Class labels.
        meta: Dataset metadata.
        filepath: Output path (e.g., "data/train.csv").
        include_labels: If True, include label columns.
        **csv_kwargs: Additional arguments for pd.DataFrame.to_csv()
                      (e.g., index=False, sep=';').

    Examples:
        >>> to_csv(X, y, meta, "output/dataset.csv", index=False)
    """
    df = to_labeled_dataframe(X, y, meta, include_labels=include_labels)
    df.to_csv(filepath, **csv_kwargs)


def to_parquet(
        X: pd.DataFrame | NDArray[np.float64],
        y: NDArray[np.int64],
        meta: DatasetMeta,
        filepath: str | Path,
        *,
        include_labels: bool = True,
        **parquet_kwargs,
) -> None:
    """Export dataset to Parquet file (efficient for large datasets).

    Args:
        X: Feature matrix.
        y: Class labels.
        meta: Dataset metadata.
        filepath: Output path (e.g., "data/train.parquet").
        include_labels: If True, include label columns.
        **parquet_kwargs: Additional arguments for pd.DataFrame.to_parquet()
                          (e.g., compression='gzip', engine='pyarrow').

    Examples:
        >>> to_parquet(X, y, meta, "output/dataset.parquet")
    """
    df = to_labeled_dataframe(X, y, meta, include_labels=include_labels)
    df.to_parquet(filepath, **parquet_kwargs)