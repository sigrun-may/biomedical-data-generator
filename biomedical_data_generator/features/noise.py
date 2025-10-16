# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Functions to add noise features to datasets."""

from collections.abc import Mapping
from typing import Any

import numpy as np


def sample_noise(
    rng: np.random.Generator, n: int, dist: str, scale: float, params: Mapping[str, Any] | None = None
) -> np.ndarray:
    """Sample n noise values from the specified distribution.

    Args:
        rng: Random number generator.
        n: Number of samples.
        dist: Distribution name ("normal", "uniform", "laplace").
        scale: Scale parameter (stddev for normal/laplace, half-width for uniform).
        params: Additional distribution-specific parameters.

    Returns:
    -------
        np.ndarray: Array of shape (n,) with sampled noise values.

    Raises:
    ------
        ValueError: If dist is unsupported or parameters are invalid.
    """
    params = dict(params or {})
    if dist == "normal":
        loc = float(params.pop("loc", 0.0))
        return rng.normal(loc=loc, scale=scale, size=n)
    elif dist == "uniform":
        low = float(params.pop("low", -scale))
        high = float(params.pop("high", scale))
        return rng.uniform(low=low, high=high, size=n)
    elif dist == "laplace":
        loc = float(params.pop("loc", 0.0))
        return rng.laplace(loc=loc, scale=scale, size=n)
    else:
        raise ValueError(f"Unsupported noise_distribution: {dist}")
