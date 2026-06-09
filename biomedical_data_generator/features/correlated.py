# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

r"""Generation of correlated feature clusters simulating pathway-like modules.

Overview
--------
This module generates *correlated Gaussian feature clusters* that can be
interpreted as simplified "pathway-like" modules (e.g., sets of co-expressed
genes or co-regulated proteins).

Each cluster is defined by:

* A correlation structure (equicorrelated or Toeplitz/AR(1)).
* A correlation strength parameter ``correlation``.
* Optionally class-specific correlation strengths to mimic activation in
  specific biological conditions (e.g., tumors vs controls).
* An anchor feature with class-specific mean shifts representing diagnostic
  strength (e.g., biomarker concentration changes).

The resulting clusters are concatenated horizontally.

Statistical model
-----------------
At the core, each cluster implements a multivariate Gaussian model:

* For a given cluster with n_features (p) and a correlation matrix
  :math:`\Sigma`, we generate samples according to

  .. math::

      x \sim \mathcal{N}_p(\mu_c, \Sigma_c),

  where :math:`\mu_c` and :math:`\Sigma_c` depend on class :math:`c`.

* Two correlation structures are supported:

  - **Equicorrelated**:
    All off-diagonal entries are equal to the correlation parameter:

    .. math::

        \Sigma_{ij} =
        \begin{cases}
            1        & i = j, \\
            \rho    & i \neq j.
        \end{cases}

    where :math:`\rho` is the correlation parameter.

  - **Toeplitz / AR(1)**:
    Correlation decays with distance:

    .. math::

        \Sigma_{ij} = \rho^{\lvert i - j \rvert}.

    where :math:`\rho` is the correlation parameter.

Anchor effects (mean channel)
-----------------------------
First-moment signal is carried by the optional ``mean_channel``. When present,
the anchor feature receives the channel's per-class mean shift:

.. math::

    \mu_{anchor, c} = \text{mean\_channel.per\_class\_effect}[c],

with absent classes receiving ``0.0`` (baseline). A proxy at block column
``j`` inherits this shift structurally: the anchor's per-class effect is
propagated as ``effect * proxy_attenuation * sigma[anchor_index, j]``, where
``sigma`` is the structural correlation matrix built from
``correlation_structure`` and that class's effective correlation (the same
correlation that samples the block). The proxy shift is therefore deterministic
and decays with structural distance from the anchor under a Toeplitz structure.

**Configuration semantics** (channel model):
  - Relevance is **derived**, never declared -- there is no declared anchor
    role. A cluster is informative iff its mean channel varies across classes
    (first moment) or its effective per-class correlation varies across classes
    (second moment, via the ``covariance_channel``).
  - No ``mean_channel`` → no class-dependent mean shift on the anchor or its
    proxies.
  - A ``mean_channel`` whose effects are equal across classes contributes no
    first-moment signal (the cluster is informative only if some channel varies).

Limitations and biological realism
----------------------------------
See module docstring for detailed discussion of simplifications.
Key points:

1. Gaussian marginals (real data is often skewed, zero-inflated)
2. Linear dependence only (no thresholds, saturation)
3. Independent clusters (no pathway crosstalk)
4. Blockwise effects (partial activation not modeled)
5. No sample-level heterogeneity (no subtypes)

Intended use
------------
Realistic enough for teaching and benchmarking, but not a fully realistic
generative model for complex omics data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from biomedical_data_generator.config import CorrClusterConfig, DatasetConfig
from biomedical_data_generator.meta import _proxy_mean_offset

__all__ = [
    "build_correlation_matrix",
    "sample_correlated_data",
    "sample_all_correlated_clusters",
]

CORRELATION_ZERO_THRESHOLD = 1e-12


# ============================================================================
# Correlation matrix construction
# ============================================================================
def build_correlation_matrix(
    n_features: int,
    correlation: float,
    structure: str = "equicorrelated",
) -> np.ndarray:
    """Build a correlation matrix with specified structure.

    Args:
        n_features: Number of features (matrix dimension).
        correlation: Correlation parameter.
        structure: Either 'equicorrelated' or 'toeplitz'.

    Returns:
        Correlation matrix of shape (n_features, n_features).

    Raises:
        ValueError: If structure is unknown or correlation is out of bounds.
    """
    if structure not in {"equicorrelated", "toeplitz"}:
        raise ValueError(f"Unknown correlation structure: {structure}")

    if n_features < 2:
        raise ValueError(f"Correlation matrix requires at least two features, got {n_features}.")

    if structure == "equicorrelated":
        lower_bound = -1.0 / (n_features - 1)
        if not (lower_bound < correlation < 1.0):
            raise ValueError(
                f"For equicorrelated with n_features={n_features}, correlation must be in "
                f"({lower_bound:.4f}, 1.0), got {correlation}."
            )
        r = np.full((n_features, n_features), correlation, dtype=np.float64)
        np.fill_diagonal(r, 1.0)
        return r

    elif structure == "toeplitz":
        if not (-1.0 < correlation < 1.0):
            raise ValueError(f"For toeplitz, correlation must be in (-1.0, 1.0), got {correlation}.")
        exponents = np.abs(np.arange(n_features)[:, None] - np.arange(n_features)[None, :])
        r = correlation**exponents
        return r.astype(np.float64)

    else:
        raise ValueError(f"Unknown correlation structure: {structure}")


def _cholesky_with_jitter(
    corr_matrix: np.ndarray,
    initial_jitter: float = 1e-10,
    growth: float = 10.0,
    max_tries: int = 8,
) -> np.ndarray:
    r"""Compute a Cholesky factor with diagonal jitter fallback.

    This helper is designed to be robust for nearly singular covariance or
    correlation matrices that may arise from extreme parameter choices or
    numerical round-off.

    The strategy is:

    1. Try plain Cholesky factorization.
    2. If it fails with ``LinAlgError``, successively add a small diagonal
       jitter ``eps * I`` and retry, increasing ``eps`` by ``growth`` after
       each failed attempt.
    3. If all attempts fail, raise a ``LinAlgError``.

    Args:
        corr_matrix: Symmetric positive (semi-)definite matrix. It is assumed
            to be theoretically positive definite for the chosen parameters.
        max_tries: Maximum number of jitter attempts after the initial
            Cholesky attempt.
        initial_jitter: Starting jitter value ``eps``.
        growth: Factor by which ``eps`` is multiplied after each failed
            attempt.

    Returns:
        Lower-triangular Cholesky factor ``L`` such that
        ``L @ L.T`` approximates ``corr_matrix`` (plus small jitter).

    Raises:
        np.linalg.LinAlgError: If Cholesky factorization fails even after all
            jitter attempts.

    Notes:
        The added jitter is intentionally small and increased only as much as
        required to obtain a numerically stable factor. This trades negligible
        perturbations of the correlation structure for robust behavior in
        ill-conditioned settings, which is typically acceptable for didactic
        simulations.
    """
    try:
        # Fast path: no jitter needed.
        return np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        pass

    jitter = float(initial_jitter)
    identity = np.eye(corr_matrix.shape[0], dtype=corr_matrix.dtype)

    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(corr_matrix + jitter * identity)
        except np.linalg.LinAlgError:
            jitter *= growth

    # Should never reach here due to raise in loop
    raise np.linalg.LinAlgError(
        "Cholesky factorization failed even after adding diagonal jitter. "
        "Check correlation parameters for near-singular configurations."
    )


def sample_correlated_data(
    n_samples: int,
    n_features: int,
    correlation: float,
    *,
    structure: str = "equicorrelated",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample correlated Gaussian data with zero mean and unit variance.

    This function generates the Gaussian core for correlated feature clusters.

    Args:
        n_samples: Number of samples to generate.
        n_features: Number of features.
        correlation: Correlation parameter.
        structure: Correlation structure ('equicorrelated' or 'toeplitz').
        rng: Random number generator. If None, creates a new one.

    Returns:
        Array of shape (n_samples, n_features) with standard normal marginals
        and specified correlation structure.

    Raises:
        ValueError: If structure is invalid or correlation out of bounds.
    """
    if rng is None:
        rng = np.random.default_rng()

    corr_matrix = build_correlation_matrix(n_features, correlation, structure)
    cholesky_matrix = _cholesky_with_jitter(corr_matrix)

    z = rng.standard_normal(size=(n_samples, n_features))
    x = z @ cholesky_matrix.T

    return x


# ============================================================================
# Anchor effect application
# ============================================================================
def _apply_mean_channel(
    block: np.ndarray,
    y: np.ndarray,
    cluster_cfg: CorrClusterConfig,
) -> None:
    """Apply the per-class anchor mean shift and its proxy propagation in-place.

    The anchor at ``anchor_index`` receives the full per-class effect. A proxy at
    block column ``j`` inherits ``effect * proxy_attenuation * sigma[anchor_index, j]``,
    where ``sigma`` is the structural correlation matrix built from
    ``correlation_structure`` and that class's effective correlation -- the same
    correlation that sampled the class's block. Using the structural correlation
    (rather than the empirical correlation of the shifted data) keeps the proxy
    shift deterministic and makes the anchor the unique carrier of the
    Bayes-optimal discriminant direction.

    Args:
        block: Cluster block of shape (n_samples, n_cluster_features). Modified in-place.
        y: Class labels of shape (n_samples,).
        cluster_cfg: Cluster configuration.
    """
    if cluster_cfg.mean_channel is None:
        return

    n_features = cluster_cfg.n_cluster_features
    anchor_index = cluster_cfg.anchor_index

    for class_value in np.unique(y):
        class_index = int(class_value)
        effect = cluster_cfg.mean_effect_for_class(class_index)
        if effect == 0.0:
            continue

        class_mask = y == class_index
        block[class_mask, anchor_index] += effect

        if n_features > 1:
            rho = cluster_cfg.effective_correlation_for_class(class_index)
            for proxy_index in range(n_features):
                if proxy_index == anchor_index:
                    continue
                block[class_mask, proxy_index] += _proxy_mean_offset(
                    anchor_per_class_offset=effect,
                    distance=abs(proxy_index - anchor_index),
                    correlation_structure=cluster_cfg.correlation_structure,
                    effective_per_class_correlation=rho,
                    proxy_attenuation=cluster_cfg.proxy_attenuation,
                )


# ============================================================================
# High-level cluster generation
# ============================================================================
def _sample_cluster_block(
    y: np.ndarray,
    cluster_cfg: CorrClusterConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one correlated block, per class, from its effective correlation.

    Each class's rows are drawn with that class's effective within-block
    correlation (the covariance channel value for the class, or the cluster's
    ``baseline_correlation`` when absent), so differential co-expression is
    expressed directly in the sampled second moments. A class whose effective
    correlation is (near) zero is drawn as independent standard normals.

    Args:
        y: Class labels as 1D array of length n_samples.
        cluster_cfg: Cluster configuration.
        rng: Random number generator.

    Returns:
        Feature block of shape (n_samples, n_cluster_features).
    """
    n_samples = len(y)
    n_features = cluster_cfg.n_cluster_features
    block = np.empty((n_samples, n_features), dtype=float)

    for class_value in np.unique(y):
        class_index = int(class_value)
        class_mask = y == class_index
        n_class = int(class_mask.sum())
        if n_class == 0:
            continue

        rho = cluster_cfg.effective_correlation_for_class(class_index)
        if abs(rho) < CORRELATION_ZERO_THRESHOLD:
            class_block = rng.standard_normal(size=(n_class, n_features))
        else:
            class_block = sample_correlated_data(
                n_class,
                n_features,
                rho,
                structure=cluster_cfg.correlation_structure,
                rng=rng,
            )

        block[class_mask, :] = class_block

    return block


def sample_all_correlated_clusters(
    cfg: DatasetConfig,
    y: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, dict[int, Any]]]:
    r"""Generate and assemble all correlated feature clusters for a dataset.

    For each cluster, a Gaussian block is sampled per class from that class's
    effective within-block correlation (the covariance channel value for the
    class, or the cluster's ``baseline_correlation`` when absent), then the mean
    channel adds the per-class anchor shift with its structurally derived proxy
    propagation. Relevance is never declared; it is derived from these channels.

    Args:
        cfg: Dataset configuration with the ``corr_clusters`` field.
        y: Class labels as a 1D NumPy array of length n_samples.
        rng: Optional random number generator. If None, creates a new one.

    Returns:
        A tuple (x_clusters, cluster_meta) where:

        * x_clusters: Array of shape (n_samples, n_corr_features) with the
          assembled correlated blocks including channel effects.
        * cluster_meta: Dictionary with cluster-level metadata, keyed by field
          name then cluster id:
            - "mean_per_class_effect": cluster_id -> mean channel mapping or None
            - "covariance_per_class_correlation": cluster_id -> covariance mapping or None
            - "baseline_correlation": cluster_id -> structural baseline correlation
            - "label": cluster_id -> human-readable label
            - "structure": cluster_id -> correlation structure ("equicorrelated" or "toeplitz")
            - "proxy_attenuation": cluster_id -> anchor-to-proxy mean-propagation multiplier
            - "anchor_index": cluster_id -> structural anchor column within the block
    """
    if rng is None:
        rng = np.random.default_rng()

    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be a 1D array of class labels, got shape {y.shape}.")

    n_samples = int(y.shape[0])
    cluster_cfgs: list[CorrClusterConfig] = cfg.corr_clusters or []

    cluster_array_list: list[np.ndarray] = []
    for cluster_cfg in cluster_cfgs:
        block = _sample_cluster_block(y, cluster_cfg, rng)
        _apply_mean_channel(block, y, cluster_cfg)
        cluster_array_list.append(block)

    if cluster_array_list:
        x_clusters = np.hstack(cluster_array_list)
    else:
        x_clusters = np.empty((n_samples, 0), dtype=float)

    cluster_meta: dict[str, dict[int, Any]] = {
        "mean_per_class_effect": {
            cluster_id: (c.mean_channel.per_class_effect if c.mean_channel is not None else None)
            for cluster_id, c in enumerate(cluster_cfgs)
        },
        "covariance_per_class_correlation": {
            cluster_id: (c.covariance_channel.per_class_correlation if c.covariance_channel is not None else None)
            for cluster_id, c in enumerate(cluster_cfgs)
        },
        "baseline_correlation": {cluster_id: c.baseline_correlation for cluster_id, c in enumerate(cluster_cfgs)},
        "label": {cluster_id: c.label for cluster_id, c in enumerate(cluster_cfgs)},
        "structure": {cluster_id: c.correlation_structure for cluster_id, c in enumerate(cluster_cfgs)},
        "proxy_attenuation": {cluster_id: c.proxy_attenuation for cluster_id, c in enumerate(cluster_cfgs)},
        "anchor_index": {cluster_id: c.anchor_index for cluster_id, c in enumerate(cluster_cfgs)},
    }

    return x_clusters, cluster_meta
