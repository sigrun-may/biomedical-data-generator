# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Correlation analysis and seed search utilities (no plotting).

This module provides functions to compute correlation metrics,
assess correlation quality, search for random seeds that yield
desired correlation properties, and slice DataFrames by cluster.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Runtime dependencies
from biomedical_data_generator.config import CorrClusterConfig
from biomedical_data_generator.features.correlated import sample_cluster

# Type alias kept local to avoid circular imports
CorrelationStructure = Literal["equicorrelated", "toeplitz"]

__all__ = [
    # metrics & summaries
    "compute_correlation_metrics",
    "assess_correlation_quality",
    "pc1_share_from_corr",
    "pc1_share",
    "variance_partition_pc1",
    # seed search
    "find_seed_for_correlation",
    "find_seed_for_correlation_from_config",
    "find_best_seed_for_correlation",
    # cluster slicing
    "parse_cluster_id",
    "get_cluster_column_names",
    "get_cluster_frame",
    # correlation computation helpers (no plotting)
    "compute_correlation_matrix",
    "compute_correlation_matrix_for_cluster",
]


# ============================================================================
# Correlation Metrics
# ============================================================================
def compute_correlation_metrics(corr_matrix: NDArray[np.floating[Any]]) -> dict[str, float | int]:
    """Compute summary metrics from a correlation matrix.

    Args:
        corr_matrix: Square correlation matrix of shape (p, p).

    Returns:
        Dictionary with keys:
            - mean_offdiag
            - std_offdiag
            - min_offdiag
            - max_offdiag
            - range_offdiag
            - n_offdiag
    """
    corr_matrix = np.asarray(corr_matrix, dtype=float)
    if corr_matrix.ndim != 2 or corr_matrix.shape[0] != corr_matrix.shape[1]:
        raise ValueError("corr_matrix must be a square 2D array.")

    n_features = corr_matrix.shape[0]
    if n_features <= 1:
        return {
            "mean_offdiag": 1.0,
            "std_offdiag": 0.0,
            "min_offdiag": 1.0,
            "max_offdiag": 1.0,
            "range_offdiag": 0.0,
            "n_offdiag": 0,
        }
    mask = ~np.eye(n_features, dtype=bool)
    off = corr_matrix[mask]
    return {
        "mean_offdiag": float(np.mean(off)),
        "std_offdiag": float(np.std(off)),
        "min_offdiag": float(np.min(off)),
        "max_offdiag": float(np.max(off)),
        "range_offdiag": float(np.max(off) - np.min(off)),
        "n_offdiag": int(off.size),
    }


def assess_correlation_quality(
    X: NDArray[np.float64],
    rho_target: float,
    *,
    tolerance: float = 0.05,
    structure: CorrelationStructure = "equicorrelated",
) -> dict[str, float | bool | str]:
    """Assess how well the empirical correlation of X matches the target.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        rho_target: Target correlation value.
        tolerance: Acceptable deviation from target.
        structure: Correlation structure ("equicorrelated" or "toeplitz").

    Returns:
        Dictionary with keys:
            - mean_offdiag
            - std_offdiag
            - min_offdiag
            - max_offdiag
            - range_offdiag
            - n_offdiag
            - target
            - deviation_offdiag
            - within_tolerance
            - structure
    """
    C = np.corrcoef(X, rowvar=False)
    m = compute_correlation_metrics(C)
    dev = abs(m["mean_offdiag"] - rho_target)
    return {
        **m,
        "target": float(rho_target),
        "deviation_offdiag": float(dev),
        "within_tolerance": bool(dev <= tolerance),
        "structure": structure,
    }


# ============================================================================
# PC1 share (shared variance)
# ============================================================================
def pc1_share_from_corr(C: np.ndarray) -> float:
    """Compute the share of variance explained by the first principal component from a correlation matrix C.

    Args:
        C: Square correlation matrix of shape (p, p).

    Returns:
        Share of variance explained by the first principal component (float in [0, 1]).
    """
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square 2D array.")
    p = C.shape[0]
    if p == 0:
        return 0.0
    lam_max = float(np.linalg.eigvalsh(C).max())
    return float(lam_max / p)


def pc1_share(
    X: pd.DataFrame | np.ndarray,
    *,
    method: Literal["pearson", "kendall", "spearman"] = "pearson",
    rowvar: bool = False,
) -> float:
    """Compute the share of variance explained by the first principal component from data X.

    Args:
        X: Data matrix (DataFrame or 2D array).
        method: Correlation method ("pearson", "kendall", or "spearman").
        rowvar: If True, rows represent variables (features), otherwise columns do.

    Returns:
        Share of variance explained by the first principal component (float in [0, 1]).
    """
    if isinstance(X, pd.DataFrame):
        if method not in {"pearson", "spearman", "kendall"}:
            raise ValueError("method must be 'pearson', 'kendall' or 'spearman'")
        C = X.corr(method=method).to_numpy(dtype=float)
    else:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if method == "pearson":
            C = np.corrcoef(X, rowvar=rowvar)
        elif method == "spearman":
            ranks = pd.DataFrame(X if not rowvar else X.T).rank(axis=0)
            C = np.corrcoef(ranks.to_numpy(), rowvar=False if not rowvar else True)
        else:
            raise ValueError("method must be 'pearson' or 'spearman'")
    return pc1_share_from_corr(C)


def variance_partition_pc1(
    X: pd.DataFrame | np.ndarray,
    *,
    method: Literal["pearson", "kendall", "spearman"] = "pearson",
    rowvar: bool = False,
) -> dict:
    """Compute variance partitioning summary based on PC1 share.

    Args:
        X: Data matrix (DataFrame or 2D array).
        method: Correlation method ("pearson", "kendall", or "spearman").
        rowvar: If True, rows represent variables (features), otherwise columns do.

    Returns:
        Dictionary with keys:
            - n_features
            - pc1_evr
            - pc1_var_ratio
    """
    if isinstance(X, pd.DataFrame):
        n_features = X.shape[1] if not rowvar else X.shape[0]
    else:
        X = np.asarray(X)
        n_features = X.shape[1] if not rowvar else X.shape[0]
    evr = pc1_share(X, method=method, rowvar=rowvar)
    return {"n_features": int(n_features), "pc1_evr": float(evr), "pc1_var_ratio": float(evr)}


# ============================================================================
# Seed search (core)
# ============================================================================
def _validate_rho(structure: CorrelationStructure, p: int, rho: float) -> None:
    if structure == "equicorrelated":
        if p < 2:
            if not (-1.0 < rho < 1.0):
                raise ValueError("For p=1 require |rho| < 1.")
            return
        lower = -1.0 / (p - 1)
        if not (lower < rho < 1.0):
            raise ValueError(f"Equicorrelated requires {lower:.6f} < rho < 1.0 (p={p}), got {rho}.")
    else:
        if not (-1.0 < rho < 1.0):
            raise ValueError("Toeplitz requires |rho| < 1.0.")


def find_seed_for_correlation(
    n_samples: int,
    n_cluster_features: int,
    rho: float,
    structure: CorrelationStructure = "equicorrelated",
    *,
    metric: Literal["mean_offdiag", "min_offdiag", "max_offdiag", "std_offdiag"] = "mean_offdiag",
    tolerance: float | None = 0.02,
    threshold: float | None = None,
    op: Literal[">=", "<="] = ">=",
    start_seed: int = 0,
    max_tries: int = 200,
    return_best_on_fail: bool = True,
    return_matrix: bool = False,
    enforce_p_le_n_in_tolerance: bool = True,
) -> tuple[int, dict[str, Any]]:
    """Find a random seed that yields a cluster with desired correlation properties.

    Args:
        n_samples: Number of samples to generate.
        n_cluster_features: Number of features in the cluster.
        rho: Target correlation value.
        structure: Correlation structure ("equicorrelated" or "toeplitz").
        metric: Correlation metric to evaluate ("mean_offdiag", "min_offdiag", "max_offdiag", "std_offdiag").
        tolerance: Acceptable deviation from target (for "tolerance" mode).
        threshold: Metric threshold (for "threshold" mode).
        op: Operator for threshold comparison (">=" or "<=").
        start_seed: Seed to start searching from.
        max_tries: Maximum number of seeds to try.
        return_best_on_fail: If True, return best found seed if none satisfy criterion.
        return_matrix: If True, include correlation matrix in metadata.
        enforce_p_le_n_in_tolerance: If True, enforce n_features <= n_samples in tolerance mode.

    Returns:
        Tuple of (seed, metadata dictionary).
    """
    if tolerance is None and threshold is None:
        raise ValueError("Provide either `tolerance` or `threshold`.")
    if n_cluster_features < 2:
        raise ValueError("n_cluster_features must be >= 2 for correlation-based selection.")
    _validate_rho(structure, n_cluster_features, rho)

    def _ok(val: float, thr: float, op_: str) -> bool:
        return (val >= thr) if op_ == ">=" else (val <= thr)

    use_tol = tolerance is not None
    mode = "tolerance" if use_tol else "threshold"

    best_seed: int | None = None
    best_meta: dict[str, Any] | None = None
    best_score: tuple[float, float] | None = None

    seed = start_seed
    for try_idx in range(1, max_tries + 1):
        rng = np.random.default_rng(seed)
        X = sample_cluster(n_samples, n_cluster_features, rng, structure=structure, rho=rho)
        C: NDArray[np.float64] = np.corrcoef(X, rowvar=False).astype(np.float64, copy=False)

        m = compute_correlation_metrics(C)
        mean_off = m["mean_offdiag"]
        deviation = abs(mean_off - rho)
        metric_val = float(m[metric])

        p_gt_n_warn = False
        if use_tol:
            if enforce_p_le_n_in_tolerance and (n_cluster_features > n_samples):
                p_gt_n_warn = True
                accepted = False
                primary, secondary = deviation, 0.0
            else:
                accepted = deviation <= float(tolerance)  # type: ignore[arg-type]
                primary, secondary = deviation, 0.0
        else:
            if threshold is None:
                raise ValueError("Threshold mode selected but no `threshold` provided.")
            accepted = _ok(metric_val, float(threshold), op)
            gap = max(0.0, float(threshold) - metric_val) if op == ">=" else max(0.0, metric_val - float(threshold))
            primary, secondary = gap, abs(metric_val - float(threshold))

        meta: dict[str, Any] = {
            "seed": seed,
            "tries": try_idx,
            "accepted": bool(accepted),
            "mode": mode,
            "structure": structure,
            "n_samples": n_samples,
            "n_features": n_cluster_features,
            "target": float(rho),
            "metric": metric,
            "metric_value": metric_val,
            "deviation_offdiag": float(deviation),
            "mean_offdiag": float(mean_off),
            "min_offdiag": float(m["min_offdiag"]),
            "max_offdiag": float(m["max_offdiag"]),
            "std_offdiag": float(m["std_offdiag"]),
            "range_offdiag": float(m["range_offdiag"]),
            "n_offdiag": int(m["n_offdiag"]),
            "tolerance": None if tolerance is None else float(tolerance),
            "threshold": None if threshold is None else float(threshold),
            "op": op,
            "p_gt_n_tolerance_warning": bool(p_gt_n_warn),
        }
        if return_matrix:
            meta["corr_matrix"] = C

        if accepted:
            return seed, meta

        score = (float(primary), float(secondary))
        if (best_score is None) or (score < best_score):
            best_score, best_seed, best_meta = score, seed, meta

        seed += 1

    if return_best_on_fail and best_seed is not None and best_meta is not None:
        best_meta = dict(best_meta)
        best_meta["accepted"] = False
        best_meta["tries"] = max_tries
        return best_seed, best_meta

    raise RuntimeError(
        f"No seed satisfied the criterion within {max_tries} tries "
        f"(mode={mode}, metric={metric}, target={rho}, tolerance={tolerance}, "
        f"threshold={threshold}, op={op})."
    )


# ============================================================================
# Convenience wrapper (CorrCluster)
# ============================================================================
def find_seed_for_correlation_from_config(
    cluster: CorrClusterConfig,
    *,
    n_samples: int,
    class_idx: int | None = None,
    metric: Literal["mean_offdiag", "min_offdiag", "max_offdiag", "std_offdiag"] = "mean_offdiag",
    tolerance: float | None = 0.02,
    threshold: float | None = None,
    op: Literal[">=", "<="] = ">=",
    start_seed: int = 0,
    max_tries: int = 200,
    return_best_on_fail: bool = True,
    return_matrix: bool = False,
    enforce_p_le_n_in_tolerance: bool = True,
) -> tuple[int, dict[str, Any]]:
    """Find a random seed for a CorrCluster that yields desired correlation properties.

    Args:
        cluster: CorrCluster configuration.
        n_samples: Number of samples to generate.
        class_idx: Optional class index for class-specific clusters.
        metric: Correlation metric to evaluate ("mean_offdiag", "min_offdiag", "max_offdiag", "std_offdiag").
        tolerance: Acceptable deviation from target (for "tolerance" mode).
        threshold: Metric threshold (for "threshold" mode).
        op: Operator for threshold comparison (">=" or "<=").
        start_seed: Seed to start searching from.
        max_tries: Maximum number of seeds to try.
        return_best_on_fail: If True, return best found seed if none satisfy criterion.
        return_matrix: If True, include correlation matrix in metadata.
        enforce_p_le_n_in_tolerance: If True, enforce n_features <= n_samples in tolerance mode.

    Returns:
        Tuple of (seed, metadata dictionary).
    """
    if class_idx is None or not cluster.is_class_specific():
        rho = cluster.rho
        structure = cluster.structure
    else:
        rho = cluster.get_rho_for_class(class_idx)
        structure = cluster.get_structure_for_class(class_idx)

    seed, meta = find_seed_for_correlation(
        n_samples=n_samples,
        n_cluster_features=cluster.n_cluster_features,
        rho=rho,
        structure=structure,
        metric=metric,
        tolerance=tolerance,
        threshold=threshold,
        op=op,
        start_seed=start_seed,
        max_tries=max_tries,
        return_best_on_fail=return_best_on_fail,
        return_matrix=return_matrix,
        enforce_p_le_n_in_tolerance=enforce_p_le_n_in_tolerance,
    )

    meta = dict(meta)
    meta.update(
        {
            "cluster_label": cluster.label,
            "cluster_anchor_role": cluster.anchor_role,
            "cluster_random_state": cluster.random_state,
            "class_idx": class_idx,
        }
    )
    return seed, meta


# ============================================================================
# Best-of-N
# ============================================================================
def find_best_seed_for_correlation(
    max_tries: int,
    n_samples: int,
    n_cluster_features: int,
    rho: float,
    structure: CorrelationStructure = "equicorrelated",
    *,
    start_seed: int = 0,
) -> tuple[int, dict[str, float]]:
    """Find the seed that yields the closest empirical correlation to the target over n_trials.

    Args:
        max_tries: Number of random seeds to try.
        n_samples: Number of samples to generate.
        n_cluster_features: Number of features in the cluster.
        rho: Target correlation value.
        structure: Correlation structure ("equicorrelated" or "toeplitz").
        start_seed: Seed to start searching from.

    Returns:
        Tuple of (best_seed, metrics dictionary).
    """
    _validate_rho(structure, n_cluster_features, rho)
    best_seed = start_seed
    best_delta = float("inf")
    best_metrics: dict[str, float] | None = None

    for s in range(start_seed, start_seed + max_tries):
        rng = np.random.default_rng(s)
        X = sample_cluster(n_samples, n_cluster_features, rng, structure=structure, rho=rho)
        C = np.corrcoef(X, rowvar=False)
        m = compute_correlation_metrics(C)
        delta = abs(m["mean_offdiag"] - rho)
        if delta < best_delta:
            best_seed, best_delta, best_metrics = s, delta, m

    assert best_metrics is not None
    return best_seed, {**best_metrics, "delta_offdiag": float(best_delta)}


# ============================================================================
# Cluster slicing helpers (no plotting)
# ============================================================================
def parse_cluster_id(name: str, prefix_corr: str = "corr") -> int | None:
    """Parse cluster ID from column name like 'corr3_anchor' or 'corr2_5'.

    Args:
        name: Column name string.
        prefix_corr: Prefix indicating correlated features.

    Returns:
        Cluster ID as integer, or None if not matching.
    """
    m = re.match(rf"^{re.escape(prefix_corr)}(\d+)_(?:anchor|\d+)$", name)
    return int(m.group(1)) if m else None


def _natural_member_sort(names: Sequence[str]) -> list[str]:
    def key(name: str) -> tuple[int, str]:
        m = re.search(r"_(\d+)$", str(name))
        return (int(m.group(1)) if m else 10**9, str(name))

    return sorted(names, key=key)


def get_cluster_column_names(
    df: pd.DataFrame,
    meta: Any,  # must provide: meta.corr_cluster_indices, meta.anchor_idx
    cluster_id: int,
    *,
    anchor_first: bool = True,
    natural_sort_rest: bool = True,
) -> list[str]:
    """Get column names for a given cluster ID from DataFrame and metadata.

    Args:
        df: DataFrame with all features.
        meta: Metadata object with cluster information.
        cluster_id: ID of the cluster to extract.
        anchor_first: If True, anchor feature is placed first (if available).
        natural_sort_rest: If True, non-anchor features are sorted naturally.

    Returns:
        List of column names in the specified order.
    """
    cluster_map: dict[int, list[int]] = meta.corr_cluster_indices
    cols_idx: list[int] = list(cluster_map[cluster_id])
    names = [str(df.columns[i]) for i in cols_idx]

    if not anchor_first:
        return _natural_member_sort(names) if natural_sort_rest else names

    anchor_idx_map: dict[int, int | None] = meta.anchor_idx
    anchor_index = anchor_idx_map.get(cluster_id, None)
    if anchor_index is not None:
        anchor_name = str(df.columns[anchor_index])
        if anchor_name in names:
            rest = [n for n in names if n != anchor_name]
            return [anchor_name] + (_natural_member_sort(rest) if natural_sort_rest else rest)

    return _natural_member_sort(names) if natural_sort_rest else names


def get_cluster_frame(
    df: pd.DataFrame,
    meta: Any,
    cluster_id: int,
    *,
    anchor_first: bool = True,
    natural_sort_rest: bool = True,
) -> pd.DataFrame:
    """Get DataFrame slice for a given cluster ID.

    Args:
        df: DataFrame with all features.
        meta: Metadata object with cluster information.
        cluster_id: ID of the cluster to extract.
        anchor_first: If True, anchor feature is placed first (if available).
        natural_sort_rest: If True, non-anchor features are sorted naturally.

    Returns:
        DataFrame slice with columns for the specified cluster.
    """
    cols = get_cluster_column_names(
        df, meta, cluster_id, anchor_first=anchor_first, natural_sort_rest=natural_sort_rest
    )
    return df.loc[:, cols]


# ============================================================================
# Correlation computation helpers (no plotting)
# ============================================================================
def compute_correlation_matrix(
    df_like: pd.DataFrame,
    *,
    method: Literal["pearson", "kendall", "spearman"] = "spearman",
) -> tuple[NDArray[np.float64], list[str]]:
    """Compute the correlation matrix from a DataFrame-like object.

    Args:
        df_like: DataFrame-like object with features as columns.
        method: Correlation method ("pearson", "kendall", or "spearman").

    Returns:
        Tuple of (correlation matrix as 2D NumPy array, list of column labels).
    """
    C_df = df_like.corr(method=method)
    C: NDArray[np.float64] = np.asarray(C_df.to_numpy(dtype=float), dtype=np.float64)
    labels = [str(c) for c in df_like.columns]
    return C, labels


def compute_correlation_matrix_for_cluster(
    df: pd.DataFrame,
    meta: Any,
    cluster_id: int,
    *,
    method: Literal["pearson", "kendall", "spearman"] = "spearman",
    anchor_first: bool = True,
    natural_sort_rest: bool = True,
) -> tuple[NDArray[np.float64], list[str]]:
    """Compute the correlation matrix for a specific cluster in the DataFrame.

    Args:
        df: DataFrame with all features.
        meta: Metadata object with cluster information.
        cluster_id: ID of the cluster to extract.
        method: Correlation method ("pearson", "kendall", or "spearman").
        anchor_first: If True, anchor feature is placed
            first (if available).
        natural_sort_rest: If True, non-anchor features are sorted naturally.

    Returns:
        Tuple of (correlation matrix as 2D NumPy array, list of column labels
            for the specified cluster).
    """
    df_c = get_cluster_frame(df, meta, cluster_id, anchor_first=anchor_first, natural_sort_rest=natural_sort_rest)
    return compute_correlation_matrix(df_c, method=method)
