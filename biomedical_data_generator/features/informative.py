# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Generation of standalone informative features and class separation.

This module builds numeric class labels from ``DatasetConfig.class_configs``,
samples base values for standalone informative features according to per-class
distributions, and applies class-wise mean shifts controlled by the
per-group ``class_sep`` on ``DatasetConfig.standalone_informative_groups``.

Scope:
    Only *standalone* informative features are produced here (i.e. informative
    features that are not cluster anchors). Correlated clusters, including
    their anchors and the attenuated proxy shifts, are handled in
    ``correlated.py``. Independent noise features are sampled directly in
    ``generator.py`` via ``utils.sampling.sample_distribution``.

The standalone-informative block is partitioned into contiguous groups, each
carrying its own ``class_sep``. The class-wise offsets are derived per group
from ``class_sep`` by ``_class_offsets_from_sep`` and applied as pure mean
shifts, so the per-class distribution shape is preserved.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from biomedical_data_generator.config import DatasetConfig
from biomedical_data_generator.utils.sampling import sample_distribution

__all__ = [
    "generate_informative_features",
    "resolve_standalone_groups",
]


def _resolve_group_sep(class_sep: float | Sequence[float], n_classes: int) -> list[float]:
    """Resolve a group's ``class_sep`` to a length ``n_classes - 1`` vector.

    A scalar broadcasts to ``n_classes - 1`` equal pairwise separations; a
    sequence is returned as-is (its length is validated on ``DatasetConfig``).

    Args:
        class_sep: Scalar separation or an explicit sequence of pairwise
            separations.
        n_classes: Number of classes.

    Returns:
        list[float]: Pairwise separations of length ``n_classes - 1``.
    """
    if isinstance(class_sep, Sequence):
        return [float(s) for s in class_sep]
    return [float(class_sep)] * (n_classes - 1)


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


def resolve_standalone_groups(cfg: DatasetConfig) -> list[tuple[tuple[int, ...], tuple[float, ...]]]:
    """Resolve the column layout and centered per-class offsets of each group.

    Mirrors exactly the contiguous, declaration-order layout and the per-class
    mean shifts applied by :func:`generate_informative_features`: group ``g``
    occupies the next ``n_features`` columns at the front of the matrix, and its
    offset vector (length ``n_classes``) is derived from the group's ``class_sep``
    by the same ``_resolve_group_sep`` / ``_class_offsets_from_sep`` path. Because
    every value is read from the resolved config, the result is reproducible
    without the feature matrix.

    Args:
        cfg: Resolved DatasetConfig.

    Returns:
        One ``(column_indices, per_class_offset)`` pair per group in
        ``cfg.standalone_informative_groups``, in declaration order.
    """
    groups: list[tuple[tuple[int, ...], tuple[float, ...]]] = []
    col = 0
    for group in cfg.standalone_informative_groups:
        col_stop = col + int(group.n_features)
        column_indices = tuple(range(col, col_stop))
        offsets = _class_offsets_from_sep(_resolve_group_sep(group.class_sep, cfg.n_classes))
        groups.append((column_indices, tuple(float(o) for o in offsets)))
        col = col_stop
    return groups


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
    """Generate all standalone informative features (no anchors, no clusters).

    The function performs:

    1. Build numeric labels ``y`` from ``cfg.class_configs``.
    2. Allocate a matrix ``x_informative`` of shape
       ``(n_samples, n_standalone_informative)``.
    3. **Base draw** over the *entire* block: for each class in declaration
       order, sample a contiguous ``(n_class_samples, n_standalone_informative)``
       block spanning all columns.
    4. **Per-group offsets**: partition the columns into the contiguous groups of
       ``cfg.standalone_informative_groups`` (declaration order, laid out at the
       front of the matrix), and for each group add the class-wise offsets
       derived from that group's ``class_sep`` to its column range.

    Draw/offset order (load-bearing for reproducibility):
        The base draw is performed **class-by-class over the full block width**,
        identical to the pre-groups implementation. Because offsets are pure mean
        shifts that consume no randomness, the RNG draw count and order are a pure
        function of ``(config, seed)`` and are *independent of how the block is
        partitioned into groups*. For a single group spanning the whole block this
        reproduces the previous behavior byte-for-byte: the only difference from
        the old code is that the constant offset is added after the draw rather
        than inside the per-class loop, which cannot change RNG consumption.

    Args:
        cfg: DatasetConfig with validated fields and derived quantities.
        rng: NumPy random Generator.

    Returns:
        tuple:
            x_informative: Array of shape (n_samples, n_standalone_informative) with
                standalone informative features.
            y: Array of shape (n_samples,) with class labels in {0, ..., K-1}.
    """
    n_samples = cfg.n_samples
    n_inf = int(cfg.n_standalone_informative)  # standalone informative features only

    # build numeric class labels
    y = _build_class_labels(cfg)

    # Corner case: no informative features → return empty matrix
    if n_inf == 0:
        x_empty_empty = np.empty((n_samples, 0), dtype=float)
        return x_empty_empty, y

    x_informative = np.empty((n_samples, n_inf), dtype=float)

    # --- Base draw over the entire standalone-informative block ---------------
    # One draw per class spanning all columns, in class-declaration order. This
    # keeps the RNG consumption a pure function of (config, seed), independent of
    # the group partition.
    start = 0
    for cls_cfg in cfg.class_configs:
        n_cls = cls_cfg.n_samples
        stop = start + n_cls
        x_informative[start:stop, :] = sample_distribution(
            distribution=cls_cfg.class_distribution,
            params=cls_cfg.class_distribution_params,
            rng=rng,
            size=(n_cls, n_inf),
        )
        start = stop

    # --- Per-group class-wise mean shifts ------------------------------------
    # Groups are contiguous column ranges in declaration order. Each group uses
    # its own class_sep to derive centered offsets; offsets[k] is added to the
    # group's columns for every sample of class k.
    col = 0
    for group in cfg.standalone_informative_groups:
        col_stop = col + int(group.n_features)
        offsets = _class_offsets_from_sep(_resolve_group_sep(group.class_sep, cfg.n_classes))

        row = 0
        for class_index, cls_cfg in enumerate(cfg.class_configs):
            row_stop = row + cls_cfg.n_samples
            x_informative[row:row_stop, col:col_stop] += offsets[class_index]
            row = row_stop
        col = col_stop

    return x_informative, y
