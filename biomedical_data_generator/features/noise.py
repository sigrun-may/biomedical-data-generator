# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Generation of free (independent) noise features.

This module samples the pure noise block of the dataset: independent features
that carry no class-discriminating signal and belong to no correlated cluster.
Values are drawn from the distribution configured via
``DatasetConfig.noise_distribution`` / ``DatasetConfig.noise_distribution_params``.

Free informative features and class separation are handled in `informative.py`.
Correlated clusters (including noise anchors and their proxies) are handled in
`correlated.py`.
"""

from __future__ import annotations

import numpy as np

from biomedical_data_generator.config import DatasetConfig
from biomedical_data_generator.utils.sampling import sample_distribution

__all__ = [
    "generate_noise_features",
]


def generate_noise_features(cfg: DatasetConfig, rng: np.random.Generator) -> np.ndarray:
    """Sample the block of free (independent) noise features.

    Only free noise features are produced here; noise anchors and their proxies
    already live inside the correlated-cluster block built by `correlated.py`.

    The function draws from the shared, global random generator and does not
    create its own. Preserving the number and order of draws is required so that
    seed-fixed dataset generation stays reproducible.

    Args:
        cfg: Dataset configuration providing the noise distribution, its
            parameters, the sample count, and the number of free noise features.
        rng: Shared NumPy random generator used across the generation pipeline.

    Returns:
        np.ndarray: Array of shape ``(cfg.n_samples, cfg.n_noise_free)`` with the
        sampled noise features. The second dimension is zero when no free noise
        features are configured.
    """
    return sample_distribution(
        distribution=cfg.noise_distribution,
        params=cfg.noise_distribution_params,
        rng=rng,
        size=(cfg.n_samples, cfg.n_noise_free),
    )
