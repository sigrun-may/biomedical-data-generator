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

* For a given cluster with ``p`` features and a correlation matrix
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

Anchor effects (mean shifts)
-----------------------------
When ``anchor_role="informative"`` and ``anchor_effect_size`` is specified,
the anchor feature receives a class-specific mean shift:

.. math::

    \mu_{anchor, c} = \text{anchor_effect_size} \cdot \mathbb{1}_{c = anchor\_class}.

Proxy features inherit this shift through correlation but with attenuated
magnitude proportional to their correlation with the anchor.

**Configuration semantics** (enforced by CorrClusterConfig validation):
  - ``anchor_role="noise"`` → no mean shift (effect_size ignored if present)
  - ``anchor_role="informative"`` → MUST have anchor_effect_size > 0

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

import numpy as np

from biomedical_data_generator.config import CorrClusterConfig, DatasetConfig

__all__ = [
    "build_correlation_matrix",
    "sample_correlated_data",
    "apply_anchor_effects",
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
        lower_bound = -1.0 / (n_features - 1) if n_features > 1 else -1.0
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
def apply_anchor_effects(
    x: np.ndarray,
    y: np.ndarray,
    cluster_configs: list[CorrClusterConfig],
) -> np.ndarray:
    """Apply class-specific mean shifts to anchor features.

    This function modifies the data matrix in-place by adding mean shifts to
    anchor features based on their configured effect sizes and target classes.

    The anchor feature (typically the first feature in each cluster) receives
    the full effect size, while correlated proxy features receive attenuated
    shifts proportional to their empirical correlation with the anchor.

    **Effect application logic**:
      - anchor_role="noise" → no shift (effect_size ignored)
      - anchor_role="informative" + anchor_effect_size > 0 → apply shift
      - Due to CorrClusterConfig validation, informative anchors always have
        anchor_effect_size != None

    Args:
        x: Feature matrix of shape (n_samples, n_features). Modified in-place.
        y: Class labels of shape (n_samples,).
        cluster_configs: List of cluster configurations with anchor metadata.

    Returns:
        The modified feature matrix (same object as input x).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    feature_offset = 0
    for cluster_cfg in cluster_configs:
        n_cluster_features = cluster_cfg.n_cluster_features
        cluster_slice = slice(feature_offset, feature_offset + n_cluster_features)

        # Resolve numeric effect size (returns 0.0 for noise anchors)
        effect_size = cluster_cfg.resolve_anchor_effect_size()
        target_class = cluster_cfg.anchor_class

        # Skip if no effect (noise anchor or zero effect)
        if effect_size == 0.0:
            feature_offset += n_cluster_features
            continue

        # Identify samples in target class (None → all classes)
        if target_class is None:
            target_mask = np.ones(len(y), dtype=bool)
        else:
            target_mask = y == target_class

        # Apply full shift to anchor feature (first in cluster)
        anchor_idx = feature_offset
        x[target_mask, anchor_idx] += effect_size

        # Apply attenuated shifts to proxy features based on correlation
        if n_cluster_features > 1:
            cluster_data = x[:, cluster_slice]
            cluster_corr = np.corrcoef(cluster_data, rowvar=False)

            # Correlation of each proxy with anchor
            anchor_correlations = cluster_corr[0, 1:]

            for i, corr_with_anchor in enumerate(anchor_correlations, start=1):
                proxy_idx = feature_offset + i
                # Proxy shift = effect_size * correlation_with_anchor
                x[target_mask, proxy_idx] += effect_size * corr_with_anchor

        feature_offset += n_cluster_features

    return x


# ============================================================================
# High-level cluster generation
# ============================================================================
def _sample_class_specific_cluster(
    y: np.ndarray,
    n_features: int,
    cluster_cfg: CorrClusterConfig,
    structure: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a cluster with per-class correlation strengths.

    Args:
        y: Class labels as 1D array of length n_samples.
        n_features: Number of features in this cluster.
        cluster_cfg: Cluster configuration with class-specific correlations.
        structure: Correlation structure ('equicorrelated' or 'toeplitz').
        rng: Random number generator for this cluster.

    Returns:
        Feature block of shape (n_samples, n_features) with class-specific
        correlation patterns.
    """
    n_samples = len(y)
    block = np.empty((n_samples, n_features), dtype=float)

    for cls in np.unique(y):
        cls_int = int(cls)
        cls_mask = y == cls_int
        n_cls = int(cls_mask.sum())
        if n_cls == 0:
            continue

        corr_cls = float(cluster_cfg.get_correlation_for_class(cls_int))

        if abs(corr_cls) < CORRELATION_ZERO_THRESHOLD:
            cls_block = rng.standard_normal(size=(n_cls, n_features))
        else:
            cls_block = sample_correlated_data(
                n_cls,
                n_features,
                corr_cls,
                structure=structure,
                rng=rng,
            )

        block[cls_mask, :] = cls_block

    return block


def sample_all_correlated_clusters(
    cfg: DatasetConfig,
    y: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, dict[int, object]]]:
    r"""Generate and assemble all correlated feature clusters for a dataset.

    This function connects the abstract configuration with the actual data
    matrix and cluster-level metadata. It supports both global and class-specific
    correlation modes, and automatically applies anchor effects based on
    cluster configuration.

    **Anchor effect application**:
      Anchor effects are applied automatically based on cluster configuration:
      - If anchor_role="noise" → no mean shift
      - If anchor_role="informative" → mean shift applied (effect_size validated to be != None)

      No separate parameter is needed because the semantics are enforced by
      CorrClusterConfig validation.

    Args:
        cfg: Dataset configuration with corr_clusters field.
        y: Class labels as a 1D NumPy array of length n_samples.
            If None, generates labels from cfg.class_configs in sequential order.
        rng: Optional random number generator. If None, creates a new one.

    Returns:
        A tuple (x_clusters, cluster_meta) where:

        * x_clusters: Array of shape (n_samples, n_corr_features) with
          correlated clusters including anchor effects where configured.
        * cluster_meta: Dictionary with cluster-level metadata:
            - "anchor_role": cluster_id -> anchor_role
            - "anchor_effect_size": cluster_id -> effect_size
            - "anchor_class": cluster_id -> target_class
            - "label": cluster_id -> human-readable label

    Examples:
        >>> # Pure correlation (noise anchor, no mean shift)
        >>> cfg = DatasetConfig(
        ...     class_configs=[ClassConfig(50), ClassConfig(50)],
        ...     corr_clusters=[
        ...         CorrClusterConfig(
        ...             n_cluster_features=5,
        ...             correlation=0.8,
        ...             anchor_role="noise"  # No shift
        ...         )
        ...     ]
        ... )
        >>> x, meta = sample_all_correlated_clusters(cfg, rng) # y auto-generated

        >>> # Correlation + diagnostic signal (informative anchor with shift)
        >>> cfg = DatasetConfig(
        ...     class_configs=[ClassConfig(50), ClassConfig(50)],
        ...     corr_clusters=[
        ...         CorrClusterConfig(
        ...             n_cluster_features=5,
        ...             correlation=0.8,
        ...             anchor_role="informative",
        ...             anchor_effect_size="medium",  # Required for informative
        ...             anchor_class=1
        ...         )
        ...     ]
        ... )
        >>> x, meta = sample_all_correlated_clusters(cfg, rng=rng) # y auto-generated

        >>> # Advanced: provide custom labels
        >>> y_custom = np.array([...])
        >>> x, meta = sample_all_correlated_clusters(cfg, y_custom, rng)
    """
    if rng is None:
        rng = np.random.default_rng()

    if y is None:
        # Generate y from config if not provided
        labels = []
        for class_idx, class_cfg in enumerate(cfg.class_configs):
            labels.extend([class_idx] * class_cfg.n_samples)
        y = np.array(labels, dtype=np.int64)
    else:
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y must be a 1D array of class labels, got shape {y.shape}.")

    n_samples = int(y.shape[0])
    cluster_cfgs: list[CorrClusterConfig] = cfg.corr_clusters or []

    cluster_array_list: list[np.ndarray] = []

    for cluster_id, cluster_cfg in enumerate(cluster_cfgs):
        cluster_seed = int(rng.integers(0, 2**63 - 1))
        cluster_rng = np.random.default_rng(cluster_seed)
        n_features = cluster_cfg.n_cluster_features

        if not cluster_cfg.is_class_specific():
            # Global mode: one correlation value for all samples
            corr_raw = cluster_cfg.correlation

            if isinstance(corr_raw, dict):
                raise ValueError(
                    "Non class-specific CorrClusterConfig must have a scalar " "correlation, got a dict instead."
                )

            corr_global = float(corr_raw)

            if abs(corr_global) < CORRELATION_ZERO_THRESHOLD:
                block = cluster_rng.standard_normal(size=(n_samples, n_features))
            else:
                block = sample_correlated_data(
                    n_samples,
                    n_features,
                    corr_global,
                    structure=cluster_cfg.structure,
                    rng=cluster_rng,
                )

        else:
            # Class-specific mode: correlation depends on class label
            block = _sample_class_specific_cluster(
                y=y,
                n_features=n_features,
                cluster_cfg=cluster_cfg,
                structure=cluster_cfg.structure,
                rng=cluster_rng,
            )

        cluster_array_list.append(block)

    if cluster_array_list:
        x_clusters = np.hstack(cluster_array_list)
    else:
        x_clusters = np.empty((n_samples, 0), dtype=float)

    # Apply anchor effects (automatically skips noise anchors via resolve_anchor_effect_size)
    x_clusters = apply_anchor_effects(x_clusters, y, cluster_cfgs)

    # Build cluster-level metadata
    cluster_meta: dict[str, dict[int, object]] = {
        "anchor_role": {cluster_id: cfg.anchor_role for cluster_id, cfg in enumerate(cluster_cfgs)},
        "anchor_effect_size": {cluster_id: cfg.anchor_effect_size for cluster_id, cfg in enumerate(cluster_cfgs)},
        "anchor_class": {cluster_id: cfg.anchor_class for cluster_id, cfg in enumerate(cluster_cfgs)},
        "label": {cluster_id: cfg.label for cluster_id, cfg in enumerate(cluster_cfgs)},
    }

    return x_clusters, cluster_meta
