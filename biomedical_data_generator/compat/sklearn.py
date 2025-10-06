# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Sklearn-like convenience wrapper around biomedical-data-generator."""

from __future__ import annotations

from typing import Any

from biomedical_data_generator.config import DatasetConfig, NoiseDistribution
from biomedical_data_generator.generator import generate_dataset


def make_biomedical_dataset(
    n_samples: int = 30,
    n_features: int = 200,
    n_informative: int = 5,
    class_sep: float = 1.2,
    weights: tuple[float, float] | None = None,
    random_state: int | None = 42,
    # Extensions beyond sklearn:
    n_noise: int = 0,
    noise_dist: NoiseDistribution = NoiseDistribution.normal,
    *,
    return_meta: bool = False,
    return_pandas: bool = False,
) -> tuple[Any, Any] | tuple[Any, Any, object]:
    """
    Sklearn-like convenience wrapper around the biomedical-data-generator.

    Parameters mirror sklearn.make_classification where sensible; extras map to DatasetConfig.
    By default returns (X, y) as numpy arrays for broad compatibility.
    Set `return_pandas=True` to keep DataFrame/Series; set `return_meta=True` to also return meta.

    Differences vs sklearn:
    - `n_noise`/`noise_dist` are explicit.
    - Not all sklearn args are supported; this wrapper favors clarity over perfect parity.
    """
    cfg = DatasetConfig(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        class_sep=class_sep,
        weights=weights,
        random_state=random_state,
        n_noise=n_noise,
        noise_distribution=noise_dist,
    )
    X, y, meta = generate_dataset(cfg)

    if not return_pandas:
        # Convert to ndarray if possible
        try:
            import pandas as pd  # optional

            if isinstance(X, pd.DataFrame):
                X = X.values
            if hasattr(y, "values"):
                y = y.values
        except Exception:
            pass

    return (X, y, meta) if return_meta else (X, y)
