# Copyright (c) 2022 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tests for class_counts parameter in DatasetConfig and its effect on generated datasets."""


import numpy as np
import pytest
from pydantic import ValidationError

from biomedical_data_generator import DatasetConfig
from biomedical_data_generator.generator import generate_dataset


def _bincount_dict(y: np.ndarray, n_classes: int) -> dict[int, int]:
    counts = np.bincount(y, minlength=n_classes).astype(int)
    return {int(k): int(counts[k]) for k in range(n_classes)}


def test_class_counts_binary_exact():
    """Exact 80/20 split with small n; counts must match exactly and meta must reflect it."""
    cfg = DatasetConfig(
        n_samples=30,
        n_classes=2,
        class_counts={0: 24, 1: 6},
        n_informative=6,
        n_noise=10,
        class_sep=1.2,
        feature_naming="prefixed",
        random_state=42,
    )
    X, y, meta = generate_dataset(cfg, return_dataframe=True)
    assert len(y) == 30
    obs = _bincount_dict(y, cfg.n_classes)
    assert obs == {0: 24, 1: 6}
    # meta must record empirical distribution
    assert hasattr(meta, "y_counts") and dict(meta.y_counts) == obs
    assert hasattr(meta, "y_weights")
    assert meta.y_weights == (24 / 30, 6 / 30)


def test_class_counts_multiclass_exact():
    """Three-class exact allocation."""
    counts = {0: 5, 1: 3, 2: 2}
    cfg = DatasetConfig(
        n_samples=10,
        n_classes=3,
        class_counts=counts,
        n_informative=4,
        n_noise=6,
        class_sep=1.0,
        feature_naming="prefixed",
        random_state=7,
    )
    X, y, meta = generate_dataset(cfg, return_dataframe=True)
    obs = _bincount_dict(y, cfg.n_classes)
    assert obs == counts
    assert dict(meta.y_counts) == counts
    assert pytest.approx(sum(meta.y_weights), rel=0, abs=1e-12) == 1.0


def test_class_counts_precedence_over_weights():
    """If both are provided, class_counts must take precedence over weights."""
    cfg = DatasetConfig(
        n_samples=20,
        n_classes=2,
        class_counts={0: 12, 1: 8},
        weights=[0.05, 0.95],  # should be ignored
        n_informative=3,
        n_noise=5,
        class_sep=1.0,
        feature_naming="prefixed",
        random_state=123,
    )
    X, y, meta = generate_dataset(cfg, return_dataframe=True)
    obs = _bincount_dict(y, cfg.n_classes)
    assert obs == {0: 12, 1: 8}
    assert dict(meta.y_counts) == obs


def test_class_counts_allows_zero_for_a_class():
    """Zero-count for a class is allowed as long as keys cover all classes and sum matches n_samples."""
    cfg = DatasetConfig(
        n_samples=30,
        n_classes=2,
        class_counts={0: 30, 1: 0},
        n_informative=2,
        n_noise=4,
        class_sep=1.0,
        feature_naming="prefixed",
        random_state=9,
    )
    X, y, meta = generate_dataset(cfg, return_dataframe=True)
    obs = _bincount_dict(y, cfg.n_classes)
    assert obs == {0: 30, 1: 0}
    assert dict(meta.y_counts) == obs
    assert meta.y_weights == (1.0, 0.0)


def test_class_counts_validation_wrong_keys():
    """Missing or wrong keys should raise a ValidationError from DatasetConfig."""
    with pytest.raises(ValidationError):
        DatasetConfig(
            n_samples=10,
            n_classes=3,
            class_counts={0: 7, 1: 3},  # missing class 2
            n_informative=2,
            n_noise=1,
            class_sep=1.0,
            feature_naming="prefixed",
        )
    with pytest.raises(ValidationError):
        DatasetConfig(
            n_samples=10,
            n_classes=2,
            class_counts={0: 5, 2: 5},  # wrong class index 2
            n_informative=2,
            n_noise=1,
            class_sep=1.0,
            feature_naming="prefixed",
        )


def test_class_counts_validation_sum_mismatch_and_negative():
    """Sum must equal n_samples; counts must be non-negative."""
    with pytest.raises(ValidationError):
        DatasetConfig(
            n_samples=30,
            n_classes=2,
            class_counts={0: 20, 1: 20},  # sum 40 != 30
            n_informative=2,
            n_noise=3,
            class_sep=1.0,
            feature_naming="prefixed",
        )
    with pytest.raises(ValidationError):
        DatasetConfig(
            n_samples=10,
            n_classes=2,
            class_counts={0: 11, 1: -1},  # negative
            n_informative=2,
            n_noise=3,
            class_sep=1.0,
            feature_naming="prefixed",
        )


def test_class_counts_reproducibility_with_seed():
    """With fixed counts and random_state, the label order should be reproducible."""
    cfg = DatasetConfig(
        n_samples=50,
        n_classes=2,
        class_counts={0: 35, 1: 15},
        n_informative=6,
        n_noise=6,
        class_sep=1.2,
        feature_naming="prefixed",
        random_state=2025,
    )
    X1, y1, meta1 = generate_dataset(cfg, return_dataframe=True)
    X2, y2, meta2 = generate_dataset(cfg, return_dataframe=True)
    assert np.array_equal(y1, y2)
    assert dict(meta1.y_counts) == dict(meta2.y_counts) == {0: 35, 1: 15}


def test_missing_class_counts_raises():
    """Without class_counts, generation should raise ValueError."""
    cfg = DatasetConfig(
        n_samples=200,
        n_classes=2,
        n_informative=4,
        n_noise=8,
        class_sep=1.0,
        feature_naming="prefixed",
        random_state=0,
    )
    with pytest.raises(ValueError, match="class_counts must be specified"):
        generate_dataset(cfg, return_dataframe=True)
