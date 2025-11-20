# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from biomedical_data_generator.config import DistributionType  # uses the validated set of names


__all__ = ["sample_2d_array"]


def sample_2d_array(
    distribution: DistributionType,
    params: Dict[str, Any],
    rng: np.random.Generator,
    size: Tuple[int, int],
) -> np.ndarray:
    """Draw a block of independent samples from a NumPy RNG distribution.

    The `distribution` string is expected to match the corresponding
    `numpy.random.Generator` method name, except for `"exp_normal"`,
    which is implemented as `exp(rng.normal(...))`.

    All parameter names and values are validated in `DatasetConfig` /
    `ClassConfig` via `validate_distribution_params`, so this function
    only dispatches to the correct RNG method.
    """
    if distribution == "exp_normal":
        # Special case: exp of an underlying normal
        base = rng.normal(size=size, **params)
        return np.exp(base)

    try:
        fn = getattr(rng, distribution)  # e.g. rng.normal, rng.uniform, ...
    except AttributeError as exc:
        # Should not happen if DistributionType and config validators are in sync
        raise ValueError(
            f"Unsupported distribution '{distribution}' for this RNG."
        ) from exc

    return fn(size=size, **params)
