# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Correlation analysis and seed search utilities.

This module provides functions to compute correlation metrics,
assess correlation quality, search for random seeds that yield
desired correlation properties, and visualize correlation structures.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

# Runtime dependencies
from biomedical_data_generator.config import CorrCluster
from biomedical_data_generator.features.correlated import sample_cluster

# Type alias kept local to avoid circular imports
CorrelationStructure = Literal["equicorrelated", "toeplitz"]

__all__ = [
    "compute_correlation_metrics",
    "assess_correlation_quality",
    "find_seed_for_correlation",
    "find_seed_for_correlation_from_config",
    "find_best_seed_for_correlation",
    "plot_correlation_matrix",
    "plot_seed_search_history",
    "compare_structures_visually",
]


# ============================================================================
# Correlation Metrics
# ============================================================================


def compute_correlation_metrics(corr_matrix: NDArray[np.floating[Any]]) -> dict[str, float | int]:
    """Compute off-diagonal correlation statistics.

    Provides metrics useful for assessing correlation matrix quality, reporting,
    and seed selection. This is the unified entry point for all correlation metrics.

    Args:
        corr_matrix: Correlation matrix of shape (n_features, n_features).

    Returns:
        Dictionary with keys:
            - 'mean_offdiag': Mean of off-diagonal correlations
            - 'std_offdiag': Standard deviation of off-diagonal correlations
            - 'min_offdiag': Minimum off-diagonal correlation
            - 'max_offdiag': Maximum off-diagonal correlation
            - 'range_offdiag': Max - min
            - 'n_offdiag': Number of off-diagonal elements (int)

    Examples:
        >>> import numpy as np
        >>> from biomedical_data_generator.utils.correlation_tools import compute_correlation_metrics
        >>> from biomedical_data_generator.features.correlated import sample_cluster
        >>> rng = np.random.default_rng(42)
        >>> X = sample_cluster(200, 5, rng, structure="equicorrelated", rho=0.7)
        >>> C = np.corrcoef(X, rowvar=False)
        >>> metrics = compute_correlation_metrics(C)
    """
    n_features = corr_matrix.shape[0]

    # Handle edge case: single feature
    if n_features <= 1:
        return {
            "mean_offdiag": 1.0,
            "std_offdiag": 0.0,
            "min_offdiag": 1.0,
            "max_offdiag": 1.0,
            "range_offdiag": 0.0,
            "n_offdiag": 0,
        }

    # Extract off-diagonal values
    mask = ~np.eye(n_features, dtype=bool)
    off_diag_values = corr_matrix[mask]

    return {
        "mean_offdiag": float(np.mean(off_diag_values)),
        "std_offdiag": float(np.std(off_diag_values)),
        "min_offdiag": float(np.min(off_diag_values)),
        "max_offdiag": float(np.max(off_diag_values)),
        "range_offdiag": float(np.max(off_diag_values) - np.min(off_diag_values)),
        "n_offdiag": int(off_diag_values.size),
    }


def assess_correlation_quality(
    X: NDArray[np.float64],
    rho_target: float,
    *,
    tolerance: float = 0.05,  # FIXED: Renamed from tol for consistency
    structure: CorrelationStructure = "equicorrelated",
) -> dict[str, float | bool | str]:  # FIXED: Added 'str'
    """Assess how well empirical correlations match the target.

    Convenience function that computes correlation matrix and metrics in one call.
    Useful for quick quality checks during generation or seed search.

    Args:
        X: Data matrix of shape (n_samples, n_features).
        rho_target: Target correlation strength.
        tolerance: Acceptable deviation from target (default 0.05).
        structure: Expected structure type (for reference in output).

    Returns:
        Dictionary with:
            - All metrics from compute_correlation_metrics()
            - 'target': The target rho value
            - 'deviation_offdiag': Absolute difference between mean_offdiag and target
            - 'within_tolerance': Boolean indicating if deviation <= tolerance
            - 'structure': The structure type (str, for reference)

    Examples:
        >>> import numpy as np
        >>> from biomedical_data_generator.utils.correlation_tools import assess_correlation_quality
        >>> from biomedical_data_generator.features.correlated import sample_cluster
        >>> rng = np.random.default_rng(42)
        >>> X = sample_cluster(300, 6, rng, structure="equicorrelated", rho=0.65)
        >>> quality = assess_correlation_quality(X, rho_target=0.65, tolerance=0.03)
        >>> print(f"Within tolerance: {quality['within_tolerance']}, deviation: {quality['deviation_offdiag']:.4f}")
    """
    C = np.corrcoef(X, rowvar=False)
    m = compute_correlation_metrics(C)
    deviation = abs(m["mean_offdiag"] - rho_target)
    return {
        **m,
        "target": float(rho_target),
        "deviation_offdiag": float(deviation),
        "within_tolerance": bool(deviation <= tolerance),
        "structure": structure,
    }


# ============================================================================
# Seed search (core)
# ============================================================================


def _validate_rho(structure: CorrelationStructure, p: int, rho: float) -> None:
    """Validate that rho satisfies positive-definiteness constraints.

    This is a pre-check before expensive sampling; build_correlation_matrix()
    will also validate, but failing early saves computation time.

    Args:
        structure: Correlation structure type.
        p: Number of features.
        rho: Correlation strength to validate.

    Raises:
        ValueError: If rho is outside valid bounds for the given structure/p.
    """
    if structure == "equicorrelated":
        if p < 2:
            if not (-1.0 < rho < 1.0):
                raise ValueError("For p=1 require |rho| < 1.")
            return
        lower = -1.0 / (p - 1)
        if not (lower < rho < 1.0):
            raise ValueError(f"Equicorrelated requires {lower:.6f} < rho < 1.0 (p={p}), got {rho}.")
    else:  # toeplitz
        if not (-1.0 < rho < 1.0):
            raise ValueError("Toeplitz requires |rho| < 1.0.")


def find_seed_for_correlation(
    n_samples: int,
    n_cluster_features: int,
    rho_target: float,
    structure: CorrelationStructure = "equicorrelated",
    *,
    metric: Literal["mean_offdiag", "min_offdiag", "max_offdiag", "std_offdiag"] = "mean_offdiag",
    # Note: range_offdiag and n_offdiag excluded (not useful for optimization)
    tolerance: float | None = 0.02,
    threshold: float | None = None,
    op: Literal[">=", "<="] = ">=",
    start_seed: int = 0,
    max_tries: int = 200,
    return_best_on_fail: bool = True,
    return_matrix: bool = False,
    enforce_p_le_n_in_tolerance: bool = True,
) -> tuple[int, dict[str, Any]]:
    """Find a random seed that produces target correlation.

    Two search modes (tolerance takes precedence):

    1. **Tolerance mode** (default, recommended for teaching):
       Accepts when |mean_offdiag - rho_target| <= tolerance.
       Most intuitive for "get close enough" scenarios.

    2. **Threshold mode** (for optimization):
       Accepts when metric satisfies: metric {op} threshold.
       Useful for "at least X" or "at most X" requirements.

    Args:
        n_samples: Number of samples to generate for testing.
        n_cluster_features: Number of features in the cluster.
        rho_target: Target correlation strength (used for reporting).
        structure: Correlation structure ("equicorrelated" or "toeplitz").
        metric: Which statistic to optimize ("mean_offdiag", "min_offdiag", etc.).
        tolerance: Acceptable deviation from rho_target (tolerance mode).
            If None, uses threshold mode instead.
        threshold: Acceptance threshold (threshold mode). Overrides tolerance if both given.
        op: Comparison operator for threshold mode (">=" or "<=").
        start_seed: First seed to try.
        max_tries: Maximum number of seeds to test.
        return_best_on_fail: If True, returns best-found seed when strict
            criteria fails. Prevents frustrating crashes during exploration.
            Recommended: True for interactive/teaching, False for production.
        return_matrix: If True, includes actual correlation matrix in metadata.
            Useful for debugging but increases memory usage.
        enforce_p_le_n_in_tolerance: If True, auto-rejects when p > n in tolerance mode
            (correlation estimates unstable with more features than samples).

    Returns:
        Tuple of (best_seed, metadata) where metadata contains:
            - 'seed': Selected seed
            - 'accepted': True if criteria met, False if best-on-fail used
            - 'tries': Number of attempts
            - 'mean_offdiag', 'std_offdiag', 'min_offdiag', 'max_offdiag': Correlation metrics
            - 'deviation_offdiag': |mean_offdiag - rho_target|
            - 'tolerance': Tolerance value used (None if threshold mode)
            - 'threshold': Threshold value used (None if tolerance mode)
            - 'p_gt_n_tolerance_warning': True if p > n caused auto-rejection

    Raises:
        ValueError: If both tolerance and threshold are None, or if n_cluster_features < 2.
        RuntimeError: If no seed meets criteria within max_tries and return_best_on_fail=False.

    Examples:
        Find seed with mean_offdiag within 0.03 of target:

        >>> from biomedical_data_generator.utils.correlation_tools import find_seed_for_correlation
        >>> seed, meta = find_seed_for_correlation(
        ...     n_samples=200,
        ...     n_cluster_features=5,
        ...     rho_target=0.7,
        ...     tolerance=0.03,
        ...     max_tries=50
        ... )
        >>> print(f"Seed {seed}: mean={meta['mean_offdiag']:.3f}, deviation={meta['deviation_offdiag']:.4f}")

        Threshold mode (ensure min_offdiag >= 0.55):

        >>> seed, meta = find_seed_for_correlation(
        ...     n_samples=300,
        ...     n_cluster_features=6,
        ...     rho_target=0.65,
        ...     metric="min_offdiag",
        ...     threshold=0.55,
        ...     op=">=",
        ...     tolerance=None
        ... )
    """
    if tolerance is None and threshold is None:
        raise ValueError("Provide either `tolerance` (tolerance mode) or `threshold` (threshold mode).")
    if n_cluster_features < 2:
        raise ValueError("n_cluster_features must be >= 2 for correlation-based selection.")

    # Pre-validate to fail fast before expensive sampling
    _validate_rho(structure, n_cluster_features, rho_target)

    use_tolerance_mode = tolerance is not None
    mode = "tolerance" if use_tolerance_mode else "threshold"

    def _ok(val: float, thr: float, op_: str) -> bool:
        if op_ == ">=":
            return val >= thr
        if op_ == "<=":
            return val <= thr
        raise ValueError(f"Invalid op: {op_!r}")

    best_seed: int | None = None
    best_meta: dict[str, Any] | None = None
    # Primary/secondary scores for deterministic tie-breaking
    best_score: tuple[float, float] | None = None

    seed = start_seed
    for try_idx in range(1, max_tries + 1):
        rng = np.random.default_rng(seed)
        X = sample_cluster(n_samples, n_cluster_features, rng, structure=structure, rho=rho_target)
        C: NDArray[np.float64] = np.corrcoef(X, rowvar=False).astype(np.float64, copy=False)

        m = compute_correlation_metrics(C)
        mean_off = m["mean_offdiag"]
        deviation = abs(mean_off - rho_target)
        metric_val = float(m[metric])

        p_gt_n_warn = False
        if use_tolerance_mode:
            # Warn if p > n (correlation estimates unreliable)
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
            # Distance to acceptance (0 if accepted)
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
            "target": float(rho_target),
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
        f"(mode={mode}, metric={metric}, target={rho_target}, tolerance={tolerance}, "
        f"threshold={threshold}, op={op})."
    )


# ============================================================================
# Convenience wrapper (CorrCluster)
# ============================================================================


def find_seed_for_correlation_from_config(
    cluster: CorrCluster,
    *,
    n_samples: int,
    class_idx: int | None = None,
    metric: Literal["mean_offdiag", "min_offdiag", "max_offdiag", "std_offdiag"] = "mean_offdiag",
    tolerance: float | None = 0.02,  # FIXED: Renamed from tol
    threshold: float | None = None,
    op: Literal[">=", "<="] = ">=",
    start_seed: int = 0,
    max_tries: int = 200,
    return_best_on_fail: bool = True,
    return_matrix: bool = False,
    enforce_p_le_n_in_tolerance: bool = True,  # FIXED: Renamed
) -> tuple[int, dict[str, Any]]:
    """Find seed for a CorrCluster (handles class-specific rho/structure).

    Thin wrapper that resolves class-specific rho/structure from a CorrCluster,
    then delegates to find_seed_for_correlation(). Saves you from manually
    extracting cluster.rho, cluster.get_rho_for_class(), etc.

    Args:
        cluster: CorrCluster instance to analyze.
        n_samples: Number of samples for testing.
        class_idx: If provided and cluster uses class_rho, uses class-specific settings.
        metric: Which statistic to optimize.
        tolerance: Tolerance mode parameter.
        threshold: Threshold mode parameter.
        op: Comparison operator for threshold mode.
        start_seed: First seed to try.
        max_tries: Maximum attempts.
        return_best_on_fail: Return best seed if strict criteria fails.
        return_matrix: Include correlation matrix in metadata.
        enforce_p_le_n_in_tolerance: Warn/reject when p > n in tolerance mode.

    Returns:
        (seed, metadata) - same as find_seed_for_correlation, plus:
            - 'cluster_label': cluster.label
            - 'cluster_anchor_role': cluster.anchor_role
            - 'cluster_random_state': cluster.random_state
            - 'class_idx': class_idx used (if any)

    Examples:
        >>> from biomedical_data_generator import CorrCluster
        >>> from biomedical_data_generator.utils.correlation_tools import find_seed_for_correlation_from_config
        >>> cluster = CorrCluster(
        ...     n_cluster_features=5,
        ...     rho=0.7,
        ...     class_rho={1: 0.9},  # Higher correlation in class 1
        ...     anchor_role="informative"
        ... )
        >>> # Find seed for class 1 (diseased patients with strong correlation)
        >>> seed, meta = find_seed_for_correlation_from_cluster(
        ...     cluster, n_samples=200, class_idx=1, tolerance=0.03
        ... )
    """
    # Resolve class-specific settings
    if class_idx is None or not cluster.is_class_specific():
        rho = cluster.rho
        structure = cluster.structure
    else:
        rho = cluster.get_rho_for_class(class_idx)
        structure = cluster.get_structure_for_class(class_idx)

    # Delegate to core search
    seed, meta = find_seed_for_correlation(
        n_samples=n_samples,
        n_cluster_features=cluster.n_cluster_features,
        rho_target=rho,
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

    # Augment metadata with cluster info
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
# Best-of-N (simple scanner; uses mean_offdiag)
# ============================================================================


def find_best_seed_for_correlation(
    n_trials: int,
    n_samples: int,
    n_features: int,
    rho: float,
    structure: CorrelationStructure = "equicorrelated",
    *,
    start_seed: int = 0,
) -> tuple[int, dict[str, float]]:
    """Try n_trials seeds and return the one minimizing |mean_offdiag - rho|.

    Simple exhaustive search - no early stopping. Useful for small-scale
    optimization when you want guaranteed best-of-N rather than first-acceptable.

    Args:
        n_trials: Number of seeds to test.
        n_samples: Number of samples per test.
        n_features: Number of features per test.
        rho: Target correlation.
        structure: Correlation structure.
        start_seed: First seed in range.

    Returns:
        (best_seed, metrics) where metrics includes:
            - All values from compute_correlation_metrics()
            - 'delta_offdiag': Absolute deviation from target

    Examples:
        >>> from biomedical_data_generator.utils.correlation_tools import find_best_seed_for_correlation
        >>> seed, metrics = find_best_seed_for_correlation(
        ...     n_trials=100, n_samples=200, n_features=5, rho=0.7
        ... )
        >>> print(f"Best seed {seed} with deviation {metrics['delta_offdiag']:.4f}")
    """
    best_seed = start_seed
    best_delta = float("inf")
    best_metrics: dict[str, float] | None = None

    for s in range(start_seed, start_seed + n_trials):
        rng = np.random.default_rng(s)
        X = sample_cluster(n_samples, n_features, rng, structure=structure, rho=rho)
        C = np.corrcoef(X, rowvar=False)
        m = compute_correlation_metrics(C)
        delta = abs(m["mean_offdiag"] - rho)
        if delta < best_delta:
            best_seed, best_delta, best_metrics = s, delta, m

    assert best_metrics is not None
    return best_seed, {**best_metrics, "delta_offdiag": float(best_delta)}


# ============================================================================
# Visualization (optional deps; offdiag naming throughout)
# ============================================================================


def plot_correlation_matrix(
    corr_matrix: NDArray[np.float64],
    *,
    title: str | None = None,
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    annot: bool = True,
    fmt: str = ".2f",
    figsize: tuple[int, int] = (8, 6),
) -> None:
    """Plot correlation matrix heatmap with annotations.

    Requires matplotlib and seaborn (optional dependencies). If not installed,
    raises ImportError with helpful message.

    Args:
        corr_matrix: Correlation matrix to visualize.
        title: Plot title (default: "Correlation Matrix").
        cmap: Colormap name (default: "RdBu_r").
        vmin: Minimum value for colormap scaling (default: -1.0).
        vmax: Maximum value for colormap scaling (default: 1.0).
        annot: Whether to annotate cells with values (default: True).
        fmt: String formatting for annotations (default: ".2f").
        figsize: Figure size in inches (default: (8, 6)).

    Raises:
        ImportError: If matplotlib or seaborn is not installed.

    Examples:
        >>> import numpy as np
        >>> from biomedical_data_generator.utils.correlation_tools import plot_correlation_matrix
        >>> from biomedical_data_generator.features.correlated import build_correlation_matrix
        >>> R = build_correlation_matrix(6, 0.7, "equicorrelated")
        >>> plot_correlation_matrix(R, title="Equicorrelated (ρ=0.7)")
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        raise ImportError(
            "Visualization requires matplotlib and seaborn. "
            "Install with: pip install biomedical-data-generator[visualization]"
        ) from e

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={"label": "Correlation"},
    )
    plt.title(title or "Correlation Matrix")
    plt.tight_layout()
    plt.show()


def plot_seed_search_history(
    seed_range: range | list[int],
    n_samples: int,
    n_cluster_features: int,
    rho_target: float,
    structure: CorrelationStructure = "equicorrelated",
    *,
    metric: Literal["mean_offdiag", "min_offdiag", "max_offdiag", "std_offdiag"] = "mean_offdiag",
    tolerance: float | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> tuple[list[int], list[float]]:
    """Visualize how correlation metrics vary across different seeds.

    Useful for understanding seed sensitivity and finding "stable" regions.
    Plots the specified metric for each seed in the range.

    Args:
        seed_range: Range or list of seeds to test.
        n_samples: Number of samples to generate for each test.
        n_cluster_features: Number of features in the cluster.
        rho_target: Target correlation strength.
        structure: Correlation structure ("equicorrelated" or "toeplitz").
        metric: Which metric to plot ("mean_offdiag", "min_offdiag", etc.).
        tolerance: Optional tolerance band to highlight on plot.
        figsize: Figure size in inches (default: (10, 6)).

    Returns:
        Tuple of (seeds, metric_values) for further analysis.

    Raises:
        ImportError: If matplotlib is not installed.

    Examples:
        >>> from biomedical_data_generator.utils.correlation_tools import plot_seed_search_history
        >>> seeds, means = plot_seed_search_history(
        ...     seed_range=range(0, 50),
        ...     n_samples=200,
        ...     n_cluster_features=5,
        ...     rho_target=0.7,
        ...     metric="mean_offdiag",
        ...     tolerance=0.05
        ... )
        >>> print(f"Found {sum(abs(m - 0.7) <= 0.05 for m in means)} seeds within tolerance")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "Visualization requires matplotlib. " "Install with: pip install biomedical-data-generator[visualization]"
        ) from e

    seeds = list(seed_range)
    vals: list[float] = []

    for s in seeds:
        rng = np.random.default_rng(s)
        X = sample_cluster(n_samples, n_cluster_features, rng, structure=structure, rho=rho_target)
        C = np.corrcoef(X, rowvar=False)
        m = compute_correlation_metrics(C)
        vals.append(float(m[metric]))

    plt.figure(figsize=figsize)
    plt.plot(seeds, vals, marker="o", linestyle="-", alpha=0.7, label=f"{metric}")
    plt.axhline(y=rho_target, linestyle="--", label=f"Target (ρ={rho_target})")
    if tolerance is not None and metric == "mean_offdiag":
        plt.axhline(y=rho_target + tolerance, linestyle=":", alpha=0.6, label="Tolerance band")
        plt.axhline(y=rho_target - tolerance, linestyle=":", alpha=0.6)
        plt.fill_between(seeds, rho_target - tolerance, rho_target + tolerance, alpha=0.1)
    plt.xlabel("Seed")
    plt.ylabel(metric)
    plt.title(f"Seed Sensitivity Analysis: {metric}\n(n={n_samples}, p={n_cluster_features}, structure={structure})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return seeds, vals


def compare_structures_visually(
    n_samples: int,
    n_cluster_features: int,
    rho: float,
    seed: int = 42,
    figsize: tuple[int, int] = (14, 6),
    annot: bool = True,
) -> None:
    """Compare equicorrelated vs. toeplitz structures side-by-side.

    Generates data with both structures and plots their empirical correlation matrices.
    Useful for teaching the difference between correlation structures.

    Args:
        n_samples: Number of samples to generate.
        n_cluster_features: Number of features in each cluster.
        rho: Correlation strength (same for both structures).
        seed: Random seed for reproducibility.
        figsize: Figure size in inches (default: (14, 6)).
        annot: Whether to annotate heatmap cells (default: True).

    Raises:
        ImportError: If matplotlib or seaborn is not installed.

    Examples:
        >>> from biomedical_data_generator.utils.correlation_tools import compare_structures_visually
        >>> compare_structures_visually(n_samples=300, n_cluster_features=6, rho=0.6)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        raise ImportError(
            "Visualization requires matplotlib and seaborn. "
            "Install with: pip install biomedical-data-generator[visualization]"
        ) from e

    rng = np.random.default_rng(seed)
    X_eq = sample_cluster(n_samples, n_cluster_features, rng, structure="equicorrelated", rho=rho)
    rng = np.random.default_rng(seed)
    X_tp = sample_cluster(n_samples, n_cluster_features, rng, structure="toeplitz", rho=rho)

    C_eq = np.corrcoef(X_eq, rowvar=False)
    C_tp = np.corrcoef(X_tp, rowvar=False)

    m_eq = compute_correlation_metrics(C_eq)
    m_tp = compute_correlation_metrics(C_tp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    sns.heatmap(
        C_eq,
        annot=annot,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax1,
        cbar_kws={"label": "Correlation"},
    )
    ax1.set_title(
        f"Equicorrelated (ρ={rho})\nMean_offdiag: "
        f"{m_eq['mean_offdiag']:.3f}, Range_offdiag: {m_eq['range_offdiag']:.3f}"
    )
    sns.heatmap(
        C_tp,
        annot=annot,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax2,
        cbar_kws={"label": "Correlation"},
    )
    ax2.set_title(
        f"Toeplitz (ρ={rho})\nMean_offdiag: {m_tp['mean_offdiag']:.3f}, Range_offdiag: {m_tp['range_offdiag']:.3f}"
    )
    plt.tight_layout()
    plt.show()
