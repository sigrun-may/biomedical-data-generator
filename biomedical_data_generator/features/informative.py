# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Generation of free informative features and class separation.

This module builds numeric class labels from DatasetConfig.class_configs,
samples base values for free informative features according to per-class
distributions, and applies class-wise mean shifts controlled by
DatasetConfig.class_sep.

Correlated clusters (including anchors) are handled in `correlated.py`.
Noise features are handled in `noise.py`. The shifting logic is implemented
in `shift_classes` and can be reused by other modules (for example, for
anchor effects in correlated clusters).
"""

from __future__ import annotations

import numpy as np

from biomedical_data_generator.config import DatasetConfig
from biomedical_data_generator.utils.sampling import sample_distribution

__all__ = [
    "generate_informative_features",
]


def _class_offsets_from_sep(sep_vec: list[float]) -> np.ndarray:
    """Construct centered class-wise offsets from a (K-1,) separation vector.

    The returned offsets have length K where K = len(sep_vec) + 1. Offsets
    are cumulative sums of the separation entries and are mean-centered.

    Args:
        sep_vec: 1-D array of length K-1 representing pairwise separations.

    Returns:
        np.ndarray: 1-D array of length K with class offsets whose mean is zero.
    """
    sep = np.asarray(sep_vec, dtype=float).ravel()
    offsets = np.concatenate(([0.0], np.cumsum(sep)))
    offsets -= offsets.mean()
    return offsets


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------
def _build_class_labels(cfg: DatasetConfig) -> np.ndarray:
    """Build numeric class labels 0..K-1 from DatasetConfig.class_configs.

    Args:
        cfg: DatasetConfig containing class_configs with per-class n_samples.

    Returns:
        np.ndarray: 1-D integer array of length cfg.n_samples with labels in
            {0, ..., K-1}.

    Raises:
        RuntimeError: If the concatenated label length does not match cfg.n_samples.
    """
    labels: list[np.ndarray] = []
    for idx, cls_cfg in enumerate(cfg.class_configs):
        labels.append(np.full(cls_cfg.n_samples, idx, dtype=int))
    y = np.concatenate(labels, axis=0)
    if y.shape[0] != cfg.n_samples:
        raise RuntimeError(f"Inconsistent label construction: got {y.shape[0]} labels, expected {cfg.n_samples}.")
    return y


# ---------------------------------------------------------------------------
# Public API: generate_informative_features
# ---------------------------------------------------------------------------
def generate_informative_features(
    cfg: DatasetConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate all free informative features (no anchors, no clusters).

    The function performs:
    1. Build numeric labels `y` from `cfg.class_configs`.
    2. Allocate a matrix `x_informative` of shape (n_samples, n_informative_free).
    3. Sample base values for each class via `sample_2d_array`.
    4. Apply class-wise mean shifts (multi-class offsets only).

    Args:
        cfg: DatasetConfig with validated fields and derived quantities.
        rng: NumPy random Generator.

    Returns:
        tuple:
            x_informative: Array of shape (n_samples, n_informative_free) with
                free informative features.
            y: Array of shape (n_samples,) with class labels in {0, ..., K-1}.
    """
    n_samples = cfg.n_samples
    n_inf = int(cfg.n_informative_free)  # excludes informative anchors by design

    # build numeric class labels
    y = _build_class_labels(cfg)

    # Corner case: no informative features → return empty matrix
    if n_inf == 0:
        x_empty_empty = np.empty((n_samples, 0), dtype=float)
        return x_empty_empty, y

    # sample base values and apply class-wise shifts
    offsets = _class_offsets_from_sep(cfg.class_sep)
    x_informative = np.empty((cfg.n_samples, n_inf), dtype=float)

    start = 0
    for cls_cfg, offset in zip(cfg.class_configs, offsets, strict=True):
        n_cls = cls_cfg.n_samples
        stop = start + n_cls

        class_samples = sample_distribution(
            distribution=cls_cfg.class_distribution,
            params=cls_cfg.class_distribution_params,
            rng=rng,
            size=(n_cls, n_inf),
        )
        # apply mean shifts to all informative columns
        class_samples += offset
        x_informative[start:stop, :] = class_samples
        start = stop

    return x_informative, y
