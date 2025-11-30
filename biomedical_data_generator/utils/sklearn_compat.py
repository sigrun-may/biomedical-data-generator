# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Sklearn-like convenience wrapper around biomedical-data-generator.

This module provides a single entry point :func:`make_biomedical_dataset`
that mimics :func:`sklearn.datasets.make_classification` while mapping
cleanly to the new :class:`DatasetConfig` / :func:`generate_dataset`
API of :mod:`biomedical_data_generator`.

The goals are:

- Familiar, scikit-learn-style signature for quick experimentation.
- A *thin* translation layer to :class:`DatasetConfig`, so that users
  can "graduate" to the full configuration model once they need more
  control.
- Numpy / pandas outputs that plug directly into scikit-learn pipelines.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from biomedical_data_generator.config import (
    BatchEffectsConfig,
    ClassConfig,
    CorrClusterConfig,
    DatasetConfig,
)
from biomedical_data_generator.generator import generate_dataset

# ---------------------------------------------------------------------------#
# Helper: translate total sample size + weights into per-class counts
# ---------------------------------------------------------------------------#


def _compute_class_sizes(
    n_samples: int,
    n_classes: int,
    weights: Sequence[float] | None,
) -> list[int]:
    """Translate total sample size + class weights into per-class counts.

    This mirrors scikit-learn semantics in a simplified way:

    - If ``weights is None``       → classes are (approximately) equally sized.
    - If ``weights`` is given      → must have length ``n_classes``.
                                     Values are normalized to sum to 1.0.

    Rounding is handled by assigning floor(n_samples * w_i) to each class
    and then distributing any remainder to the classes with the largest
    fractional parts.
    """
    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}.")

    if weights is None:
        # Equal-sized classes, remainder distributed to first classes
        base = n_samples // n_classes
        remainder = n_samples % n_classes
        return [base + (1 if i < remainder else 0) for i in range(n_classes)]

    if len(weights) != n_classes:
        raise ValueError(f"weights must have length n_classes={n_classes}, " f"got length {len(weights)}.")

    # Normalize to sum=1.0
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError(f"weights must be non-negative, got {weights}.")
    total = float(w.sum())
    if total <= 0:
        raise ValueError(f"Sum of weights must be > 0, got {total}.")
    w = w / total

    # Floor allocation + remainder distribution
    raw = w * n_samples
    counts = np.floor(raw).astype(int)
    remainder = n_samples - int(counts.sum())
    if remainder > 0:
        # Distribute remaining samples to classes with largest fractional parts
        frac = raw - counts
        for idx in np.argsort(frac)[::-1][:remainder]:
            counts[idx] += 1

    return counts.tolist()


# ---------------------------------------------------------------------------#
# Public API: sklearn-style dataset generator
# ---------------------------------------------------------------------------#


def make_biomedical_dataset(
    n_samples: int = 30,
    n_features: int = 200,
    n_informative: int = 5,
    n_redundant: int = 0,
    n_classes: int = 2,
    class_sep: float = 1.2,
    weights: tuple[float, ...] | None = None,
    random_state: int | None = 42,
    # Extensions beyond sklearn:
    n_noise: int = 0,
    noise_distribution: str = "normal",
    noise_distribution_params: dict[str, Any] | None = None,
    batch_effect: bool = False,
    n_batches: int = 1,
    batch_effect_strength: float = 0.5,
    confounding_with_class: float = 0.0,
    return_meta: bool = False,
    return_pandas: bool = False,
    **kwargs: Any,
) -> tuple[Any, Any] | tuple[Any, Any, object]:
    """Sklearn-like convenience wrapper around the biomedical-data-generator.

    Parameters broadly mirror :func:`sklearn.datasets.make_classification`
    where sensible, but are translated to the new :class:`DatasetConfig` /
    :func:`generate_dataset` design.

    Redundant features
    ------------------
    ``n_redundant`` is implemented via a single correlated feature cluster:

    - One **informative anchor** (shared signal)
    - ``n_redundant`` **proxy** features that are strongly correlated
      (equicorrelated with a high ``correlation``)

    In terms of :class:`DatasetConfig`, this means:

        n_features = n_informative + n_noise + proxies_from_clusters

    and the proxies contributed by this wrapper are exactly ``n_redundant``.

    Notes:
    -----
    - ``n_features`` must equal ``n_informative + n_redundant + n_noise``
      in this wrapper (no repeated features). If ``n_noise == 0``, it is
      inferred as ``n_features - n_informative - n_redundant``.
    - If you pass ``corr_clusters`` explicitly via ``**kwargs``, then
      ``n_redundant`` **must be 0**; you are responsible for defining the
      cluster layout yourself in that advanced mode.

    By default the function returns ``(X, y)`` using NumPy arrays for
    broad compatibility with scikit-learn. Set ``return_pandas=True`` to
    obtain a ``DataFrame`` and ``Series`` instead. Set ``return_meta=True``
    to additionally return the :class:`DatasetMeta` object.

    Returns:
    -------
    (X, y) or (X, y, meta)
        Depending on ``return_meta``. ``X`` is a NumPy array or
        pandas ``DataFrame``; ``y`` is a NumPy array or pandas ``Series``.
    """
    # ------------------------------------------------------------------
    # 0) Corr-cluster handling & feature accounting mode
    # ------------------------------------------------------------------
    explicit_corr_clusters = "corr_clusters" in kwargs and bool(kwargs["corr_clusters"])

    if explicit_corr_clusters and n_redundant > 0:
        raise ValueError(
            "n_redundant cannot be used together with an explicit "
            "'corr_clusters' configuration. Either let the sklearn-style "
            "wrapper create a redundant cluster from n_redundant, or define "
            "all CorrClusterConfig instances yourself."
        )

    if n_informative < 0 or n_redundant < 0 or n_noise < 0:
        raise ValueError(
            f"n_informative, n_redundant and n_noise must be >= 0, got "
            f"n_informative={n_informative}, n_redundant={n_redundant}, n_noise={n_noise}."
        )

    # ------------------------------------------------------------------
    # 1) Validate and resolve feature counts
    # ------------------------------------------------------------------
    if explicit_corr_clusters:
        # Advanced mode: user provides full corr_clusters; we do not try to
        # infer n_noise from n_features because we do not know the number
        # of proxies contributed by those clusters. We let DatasetConfig
        # perform consistency checks instead.
        n_noise_effective = n_noise
    else:
        # Simple sklearn-like mode: we know exactly how many proxies we add:
        # proxies_from_clusters = n_redundant (one informative anchor cluster).
        base_required = n_informative + n_redundant

        if n_features < base_required:
            raise ValueError(
                "n_features must be >= n_informative + n_redundant; "
                f"got n_features={n_features}, "
                f"n_informative={n_informative}, n_redundant={n_redundant}."
            )

        if n_noise == 0:
            # Infer remaining features as independent noise
            n_noise_effective = n_features - base_required
        else:
            n_noise_effective = n_noise
            if base_required + n_noise_effective != n_features:
                raise ValueError(
                    "In this sklearn-style wrapper we currently support only "
                    "free informative + correlated redundant (via clusters) "
                    "+ independent noise features. "
                    "Expected n_features == n_informative + n_redundant + n_noise, "
                    f"got n_features={n_features}, n_informative={n_informative}, "
                    f"n_redundant={n_redundant}, n_noise={n_noise_effective}."
                )

        if n_noise_effective < 0:
            raise ValueError(
                f"Inferred n_noise would be negative ({n_noise_effective}). "
                "Check the combination of n_features, n_informative and n_redundant."
            )

    # ------------------------------------------------------------------
    # 2) Build class configuration (sizes + labels)
    # ------------------------------------------------------------------
    class_sizes = _compute_class_sizes(
        n_samples=n_samples,
        n_classes=n_classes,
        weights=weights,
    )

    class_configs: list[ClassConfig] = [ClassConfig(n_samples=int(sz)) for sz in class_sizes]

    # ------------------------------------------------------------------
    # 3) Optional batch-effect configuration
    # ------------------------------------------------------------------
    if batch_effect and n_batches > 1:
        batch_cfg: BatchEffectsConfig | None = BatchEffectsConfig(
            n_batches=n_batches,
            effect_strength=batch_effect_strength,
            effect_type="additive",
            confounding_with_class=confounding_with_class,
            affected_features="all",  # simple wrapper: affect all features
            proportions=None,
        )
    else:
        batch_cfg = None

    # ------------------------------------------------------------------
    # 4) Optional correlated cluster for redundant features
    # ------------------------------------------------------------------
    corr_clusters: list[CorrClusterConfig] = []
    if not explicit_corr_clusters and n_redundant > 0:
        if n_informative < 1:
            raise ValueError(
                "n_redundant > 0 requires at least one informative feature "
                "to serve as the cluster anchor (n_informative >= 1)."
            )

        # One informative anchor + n_redundant proxies → total cluster size
        n_cluster_features = 1 + n_redundant

        # Strong, but not perfect, equicorrelation to represent redundancy.
        redundant_cluster = CorrClusterConfig(
            n_cluster_features=n_cluster_features,
            structure="equicorrelated",
            correlation=0.9,
            anchor_role="informative",
            anchor_effect_size=None,  # use DatasetConfig / informative defaults
            anchor_class=1 if n_classes > 1 else 0,
            random_state=None,
            label="sklearn_redundant_cluster",
        )
        corr_clusters.append(redundant_cluster)

    # ------------------------------------------------------------------
    # 5) Construct DatasetConfig
    # ------------------------------------------------------------------
    cfg_kwargs: dict[str, Any] = {
        "n_informative": int(n_informative),
        "n_noise": int(n_noise_effective),
        "class_configs": class_configs,
        "class_sep": class_sep,  # scalar → normalized by DatasetConfig validator
        "noise_distribution": noise_distribution,
    }

    if noise_distribution_params is not None:
        cfg_kwargs["noise_distribution_params"] = noise_distribution_params

    if batch_cfg is not None:
        cfg_kwargs["batch_effects"] = batch_cfg

    # First apply user kwargs (advanced mode); then we may append our cluster.
    cfg_kwargs.update(kwargs)

    # Attach automatically generated corr_clusters if we are in the simple
    # sklearn-style mode (no explicit non-empty corr_clusters from kwargs).
    if n_redundant > 0 and not explicit_corr_clusters:
        existing = cfg_kwargs.get("corr_clusters")
        if existing is None:
            # No user-provided clusters at all → use our cluster list.
            cfg_kwargs["corr_clusters"] = corr_clusters
        else:
            # User may have passed corr_clusters=None or [].
            # Treat that as "no clusters yet" and append our redundant cluster.
            if not isinstance(existing, list):
                raise TypeError(
                    "corr_clusters must be a list of CorrClusterConfig or dicts; " f"got {type(existing).__name__}."
                )
            existing_extended = list(existing)
            existing_extended.extend(corr_clusters)
            cfg_kwargs["corr_clusters"] = existing_extended

    cfg = DatasetConfig(
        random_state=random_state,
        **cfg_kwargs,
    )

    # ------------------------------------------------------------------
    # 6) Generate data via the core API
    # ------------------------------------------------------------------
    X, y, meta = generate_dataset(cfg, return_dataframe=return_pandas)

    # Convert y to Series if pandas output is requested
    if return_pandas:
        if not isinstance(X, pd.DataFrame):
            # Defensive: generate_dataset should already have returned a DataFrame
            X = pd.DataFrame(X)
        y_out: Any = pd.Series(y, name="target")
    else:
        y_out = y

    if return_meta:
        return X, y_out, meta
    return X, y_out
