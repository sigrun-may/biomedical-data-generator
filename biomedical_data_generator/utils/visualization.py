# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Plot utilities for correlation analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from numpy.typing import NDArray

from .correlation_tools import (
    compute_correlation_matrix,
    get_cluster_frame,
)

__all__ = [
    "plot_correlation_matrix",
    "plot_correlation_matrix_for_cluster",
    "plot_correlation_matrices_per_cluster",
]


# --------------------------------------------------------------------------- #
# Core: plot a correlation matrix (Matplotlib-only)
# --------------------------------------------------------------------------- #
def plot_correlation_matrix(
    correlation_matrix: NDArray[np.float64],
    *,
    title: str | None = None,
    ax: Axes | None = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    annot: bool = False,
    fmt: str = ".2f",
    labels: Sequence[str] | None = None,
    show: bool = True,
) -> tuple[Figure | SubFigure, Axes]:
    """Draw a correlation matrix as a heatmap.

    Args:
        correlation_matrix: Square correlation matrix of shape (p, p).
        title: Optional plot title.
        ax: Optional Matplotlib Axes to draw on (created if None).
        vmin: Color scale limits.
        vmax: Color scale limits.
        annot: If True, draw numeric values for small matrices (p <= 25).
        fmt: Number format for annotations.
        labels: Optional tick labels (length p). If not given, 'feature' axes labels are used.
        show: If True and a new figure is created here, call plt.show().

    Returns:
        (fig, ax): The Figure and Axes used.
    """
    correlation_matrix = np.asarray(correlation_matrix, dtype=float)
    if correlation_matrix.ndim != 2 or correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        raise ValueError("C must be a square 2D array.")

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        created = True
    else:
        fig = ax.figure  # type: ignore[assignment]

    im = ax.imshow(correlation_matrix, vmin=vmin, vmax=vmax, aspect="equal", cmap="Blues_r")
    if title:
        ax.set_title(title)

    if labels is not None:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
    else:
        ax.set_xlabel("feature")
        ax.set_ylabel("feature")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Correlation")

    if annot:
        p = correlation_matrix.shape[0]
        if p <= 25:
            for i in range(p):
                for j in range(p):
                    ax.text(j, i, format(correlation_matrix[i, j], fmt), ha="center", va="center", fontsize=7)

    if created:
        fig.tight_layout()
        if show:
            plt.show()

    return fig, ax


# --------------------------------------------------------------------------- #
# Single cluster convenience (via meta)
# --------------------------------------------------------------------------- #
def plot_correlation_matrix_for_cluster(
    df: pd.DataFrame,
    meta: Any,
    cluster_id: int,
    *,
    correlation_method: Literal["pearson", "kendall", "spearman"] = "pearson",
    anchor_first: bool = True,
    natural_sort_rest: bool = True,
    title: str | None = None,
    ax: Axes | None = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    annot: bool = False,
    fmt: str = ".2f",
    show: bool = True,
) -> NDArray[np.float64]:
    """Slice a cluster via `meta`, compute its correlation, and plot it.

    Returns the numeric correlation matrix in the plotted column order.

    Args:
        df: DataFrame with all features.
        meta: Meta object with cluster information.
        cluster_id: ID of the cluster to plot.
        correlation_method: Correlation method to use.
        anchor_first: If True, anchor features are placed first in the cluster frame.
        natural_sort_rest: If True, non-anchor features are sorted naturally.
        title: Optional plot title.
        ax: Optional Matplotlib Axes to draw on (created if None).
        vmin: Color scale limits.
        vmax: Color scale limits.
        annot: If True, draw numeric values for small matrices (p <= 25).
        fmt: Number format for annotations.
        show: If True and a new figure is created here, call plt.show().

    Returns:
        C: The computed correlation matrix as a 2D NumPy array.
    """
    # 1) Slice cluster columns (anchor first if available)
    df_block = get_cluster_frame(df, meta, cluster_id, anchor_first=anchor_first, natural_sort_rest=natural_sort_rest)

    # 2) Compute correlation (pearson/kendall/spearman)
    C, labels = compute_correlation_matrix(df_block, method=correlation_method)

    # 3) Plot
    if title is None:
        title = f"Cluster {cluster_id} — {correlation_method.capitalize()} correlation"

    plot_correlation_matrix(
        C,
        title=title,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt=fmt,
        labels=labels,
        show=show,
    )
    return C


# --------------------------------------------------------------------------- #
# Batch: one matrix per cluster (given index mapping)
# --------------------------------------------------------------------------- #
def plot_correlation_matrices_per_cluster(
    df: pd.DataFrame,
    clusters: Mapping[Any, list[int]],
    *,
    labels_map: Mapping[Any, str] | None = None,
    correlation_method: Literal["pearson", "kendall", "spearman"] = "pearson",
    vmin: float = -1.0,
    vmax: float = 1.0,
    annot: bool = False,
    fmt: str = ".2f",
    show: bool = True,
) -> dict[Any, tuple[Figure | SubFigure, Axes]]:
    """Draw one correlation matrix per cluster (cluster_id -> list of column indices).

    Args:
        df: DataFrame with all features.
        clusters: Mapping cluster_id -> list of column indices in `df`.
        labels_map: Optional mapping cluster_id -> cluster label for titles.
        correlation_method: Correlation method to use.
        vmin: Color scale limits.
        vmax: Color scale limits.
        annot: If True, draw numeric values for small matrices (p <= 25).
        fmt: Number format for annotations.
        show: If True and a new figure is created here, call plt.show().

    Returns:
        out: Mapping cluster_id -> (fig, ax) tuple for each plotted correlation matrix.

    Notes:
    -----
    - Computation is delegated to `compute_correlation_matrix` (SoC).
    - If you have a `meta` object instead of an index mapping, pass `meta.corr_cluster_indices`.
    """
    out: dict[Any, tuple[Figure | SubFigure, Axes]] = {}
    for cid, col_idx in clusters.items():
        df_block = df.iloc[:, col_idx]
        C, labels = compute_correlation_matrix(df_block, method=correlation_method)
        title = (
            labels_map.get(cid, f"Cluster {cid}") if labels_map else f"Cluster {cid}"
        ) + f" — {correlation_method.capitalize()}"

        fig_ax = plot_correlation_matrix(
            C,
            title=title,
            vmin=vmin,
            vmax=vmax,
            annot=annot,
            fmt=fmt,
            labels=labels,
            show=show,
        )
        out[cid] = fig_ax
    return out


def plot_all_correlation_clusters(
    df: pd.DataFrame,
    *,
    correlation_method: Literal["pearson", "kendall", "spearman"] = "spearman",
    title: str | None = None,
    figsize: tuple[int, int] = (10, 10),
    vmin: float = -1.0,
    vmax: float = 1.0,
    annot: bool | None = None,
    fmt: str = ".2f",
    draw_cluster_boundaries: bool = True,
    show: bool = True,
) -> tuple[Figure | SubFigure, Axes]:
    """Plot correlation matrix for all corr-cluster features with intra-cluster ordering.

    Features are sorted by (cluster_id, feature_index) so that within each cluster,
    the anchor appears first, followed by features 2, 3, 4, etc.

    Args:
        df: DataFrame with all features.
        correlation_method: Correlation method to use.
        title: Optional plot title.
        figsize: Figure size.
        vmin: Color scale minimum.
        vmax: Color scale maximum.
        annot: If True, show numeric values. If None, auto-decide based on size.
        fmt: Number format for annotations.
        draw_cluster_boundaries: If True, draw black lines between clusters.
        show: If True, call plt.show().

    Returns:
        Tuple of (figure, axes).
    """
    # 1) Select only columns containing "corr"
    corr_cols = [c for c in df.columns if "corr" in c]
    if not corr_cols:
        raise ValueError("No columns containing 'corr' found")

    df_sub = df.loc[:, corr_cols]

    # 2) Sort features: first by cluster_id, then by feature_index (anchor=1 first)
    def parse_feature_name(feature_name: str) -> tuple[int, int]:
        """Extract (cluster_id, feature_idx) from 'corrX_Y' or 'corrX_anchor'."""
        parts = feature_name.split("_")
        if len(parts) == 2 and parts[0].startswith("corr"):
            cluster_str = parts[0][4:]  # e.g., "1" from "corr1"
            if cluster_str.isdigit():
                cluster_id = int(cluster_str)
                # Treat "anchor" as index 1 (comes first)
                if parts[1] == "anchor":
                    return (cluster_id, 1)
                if parts[1].isdigit():
                    feature_idx = int(parts[1])
                    return (cluster_id, feature_idx)
        return (999, 999)  # Fallback for invalid names

    ordered_cols = sorted(corr_cols, key=parse_feature_name)
    df_ordered = df_sub[ordered_cols]

    # 3) Compute correlation
    corr = df_ordered.corr(method=correlation_method)

    # 4) Auto-decide annotation
    if annot is None:
        annot = corr.shape[0] <= 25

    # 5) Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        ax=ax,
        cmap="Blues_r",
        vmin=vmin,
        vmax=vmax,
        square=True,
        xticklabels=True,
        yticklabels=True,
        annot=annot,
        fmt=fmt if annot else "",
        cbar_kws={"label": "Correlation"},
    )

    if title is None:
        title = f"All Correlation Clusters ({correlation_method.capitalize()})"
    ax.set_title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    # 6) Draw cluster boundaries
    if draw_cluster_boundaries:

        def get_cluster_id(feature_name: str) -> int | None:
            """Extract cluster ID from 'corrX_Y' → X."""
            parts = feature_name.split("_")
            if len(parts) == 2 and parts[0].startswith("corr"):
                cluster_str = parts[0][4:]
                if cluster_str.isdigit():
                    return int(cluster_str)
            return None

        cluster_ids = [get_cluster_id(f) for f in ordered_cols]
        boundaries = [i for i in range(1, len(cluster_ids)) if cluster_ids[i] != cluster_ids[i - 1]]
        for b in boundaries:
            ax.hlines(b, *ax.get_xlim(), colors="black", linewidth=1.5)
            ax.vlines(b, *ax.get_ylim(), colors="black", linewidth=1.5)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, ax
