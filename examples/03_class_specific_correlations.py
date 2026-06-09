# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Class-specific correlations example for biomedical-data-generator.

This example demonstrates the **covariance channel** — the channel-based way to
express class-specific within-cluster correlation (differential co-expression):
- Creating feature clusters whose correlation pattern depends on the class
- Simulating pathway activation that differs between disease states
- Modeling biomarker relationships that only exist in certain conditions
- Comparing correlation matrices across classes

A ``CovarianceChannel`` maps each class index to a within-cluster correlation.
When that correlation varies across classes, the cluster is *derived-informative*
purely from its second moment — no mean shift required. Classes absent from the
mapping fall back to the cluster's ``baseline_correlation``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from biomedical_data_generator.config import (
    ClassConfig,
    CorrClusterConfig,
    CovarianceChannel,
    DatasetConfig,
    MeanChannel,
    StandaloneInformativeGroup,
)
from biomedical_data_generator.generator import generate_dataset
from biomedical_data_generator.utils.export_utils import to_csv


def compute_correlation_by_class(
    x: pd.DataFrame | np.ndarray, y: np.ndarray | pd.Series, feature_indices: list[int]
) -> dict:
    """Compute correlation matrices for specified features, grouped by class.

    Args:
        x: Feature matrix (DataFrame or ndarray) of shape (n_samples, n_features).
        y: Class labels (1-D array-like) of length n_samples.
        feature_indices: List of feature indices to include in the correlation computation.

    Returns:
        dict: Mapping from class label to correlation matrix (DataFrame).
    """
    # convert to DataFrame if needed
    if isinstance(x, np.ndarray):
        x = pd.DataFrame(x)

    correlations = {}
    for class_label in np.unique(y):
        mask = y == class_label
        x_class = x[mask].iloc[:, feature_indices]
        correlations[class_label] = x_class.corr()
    return correlations


def main() -> None:
    """Demonstrate class-specific correlation patterns in synthetic datasets."""
    print("=" * 70)
    print("Class-Specific Correlations: Simulating Context-Dependent Pathways")
    print("=" * 70)
    print()

    # Example 1: Pathway only active in diseased state
    print("Example 1: Pathway Active Only in Diseased State")
    print("-" * 70)
    print("Simulating an immune pathway where biomarkers are correlated only")
    print("in diseased patients (e.g., activated inflammatory cascade).")
    print()

    cfg1 = DatasetConfig(
        n_standalone_noise=2,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=4,
                correlation_structure="equicorrelated",
                baseline_correlation=0.0,
                # Differential co-expression: uncorrelated in healthy, strongly
                # co-expressed in diseased -> derived-informative via covariance.
                covariance_channel=CovarianceChannel(per_class_correlation={0: 0.0, 1: 0.85}),
                # A mean shift on the anchor adds a first-moment signal too.
                mean_channel=MeanChannel(per_class_effect={1: 2.0}),
                label="Disease-Activated Immune Pathway",
            )
        ],
        class_configs=[
            ClassConfig(n_samples=100, label="healthy"),
            ClassConfig(n_samples=100, label="diseased"),
        ],
        random_state=42,
    )
    x1, y1, meta1 = generate_dataset(cfg1)
    print(f"✓ Generated dataset: {x1.shape}")
    print()

    # Compute correlations per class
    cluster_id = 0  # Cluster IDs are 0-based (0, 1, 2, ...)
    cluster_features = meta1.corr_cluster_indices[cluster_id]
    correlations1 = compute_correlation_by_class(x1, y1, cluster_features)

    print("Correlation matrices for cluster features:")
    for class_label, corr_matrix in correlations1.items():
        print(f"\nClass: {class_label}")
        print(corr_matrix.round(3))

    # Example 2: Different correlation patterns in each class
    print("\n" + "=" * 70)
    print("Example 2: Different Correlation Strengths Across Classes")
    print("-" * 70)
    print("Simulating a metabolic pathway with varying coordination across")
    print("different disease subtypes.")
    print()

    cfg2 = DatasetConfig(
        # Standalone informative block with per-boundary separation.
        standalone_informative_groups=[
            StandaloneInformativeGroup(n_features=4, class_sep=[1.0, 1.5]),
        ],
        n_standalone_noise=3,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=5,
                correlation_structure="equicorrelated",
                baseline_correlation=0.0,
                # Increasing co-expression across severity -> derived-informative
                # via the covariance channel (second moment).
                covariance_channel=CovarianceChannel(per_class_correlation={0: 0.3, 1: 0.7, 2: 0.9}),
                # A graded anchor mean shift adds a first-moment signal too.
                mean_channel=MeanChannel(per_class_effect={1: 1.0, 2: 2.0}),
                label="Metabolic Dysregulation Pathway",
            )
        ],
        class_configs=[
            ClassConfig(n_samples=80, label="healthy"),
            ClassConfig(n_samples=80, label="mild_disease"),
            ClassConfig(n_samples=80, label="severe_disease"),
        ],
        random_state=42,
    )

    x2, y2, meta2 = generate_dataset(cfg2)
    print(f"✓ Generated dataset: {x2.shape}")
    print()

    # Compute correlations per class
    cluster_id = 0  # First cluster (0-based indexing)
    cluster_features2 = meta2.corr_cluster_indices[cluster_id]
    correlations2 = compute_correlation_by_class(x2, y2, cluster_features2)

    print("Mean pairwise correlations per class:")
    for class_label, corr_matrix in correlations2.items():
        # Get upper triangle (exclude diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        mean_corr = corr_matrix.values[mask].mean()
        print(f"  {class_label}: {mean_corr:.3f}")
    print()

    # Example 3: Multiple clusters with different class-specific patterns
    print("=" * 70)
    print("Example 3: Multiple Pathways with Independent Class Patterns")
    print("-" * 70)
    print("Simulating two biological pathways:")
    print("  - Pathway A: active in class 0")
    print("  - Pathway B: active in class 1")
    print()

    cfg3 = DatasetConfig(
        standalone_informative_groups=[
            StandaloneInformativeGroup(n_features=6, class_sep=[2.0]),
        ],
        n_standalone_noise=4,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                correlation_structure="equicorrelated",
                baseline_correlation=0.0,
                # Pathway A co-expressed only in class 0, with a class-0 mean shift.
                covariance_channel=CovarianceChannel(per_class_correlation={0: 0.8, 1: 0.0}),
                mean_channel=MeanChannel(per_class_effect={0: 1.5}),
                label="Pathway A (Class 0 Specific)",
            ),
            CorrClusterConfig(
                n_cluster_features=3,
                correlation_structure="equicorrelated",
                baseline_correlation=0.0,
                # Pathway B co-expressed only in class 1, with a class-1 mean shift.
                covariance_channel=CovarianceChannel(per_class_correlation={0: 0.0, 1: 0.8}),
                mean_channel=MeanChannel(per_class_effect={1: 1.5}),
                label="Pathway B (Class 1 Specific)",
            ),
        ],
        class_configs=[
            ClassConfig(n_samples=100, label="subtype_A"),
            ClassConfig(n_samples=100, label="subtype_B"),
        ],
        random_state=42,
    )

    X3, y3, meta3 = generate_dataset(cfg3)
    print(f"✓ Generated dataset: {X3.shape}")
    print()

    print("Pathway-specific correlations:")
    for cluster_id in [0, 1]:  # Cluster IDs are 0-based
        cluster_features3 = meta3.corr_cluster_indices[cluster_id]
        correlations3 = compute_correlation_by_class(X3, y3, cluster_features3)

        print(f"\nCluster {cluster_id} ({cfg3.corr_clusters[cluster_id].label}):")
        for class_label, corr_matrix in correlations3.items():
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            mean_corr = corr_matrix.values[mask].mean()
            print(f"  {class_label}: mean correlation = {mean_corr:.3f}")

    # Visualization
    print("\n" + "=" * 70)
    print("Generating Visualizations...")
    print("-" * 70)

    try:
        # Visualize Example 1: Correlation heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, (class_label, corr_matrix) in zip(axes, correlations1.items()):
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                center=0,
                square=True,
                ax=ax,
                cbar_kws={"label": "Correlation"},
            )
            ax.set_title(f"Class: {class_label}")

        plt.suptitle("Example 1: Disease-Activated Pathway", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig("class_specific_correlations_example1.png", dpi=150, bbox_inches="tight")
        print("✓ Saved Example 1 visualization to class_specific_correlations_example1.png")

        # Visualize Example 2: Compare correlation strengths
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, (class_label, corr_matrix) in zip(axes, correlations2.items()):
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                center=0,
                square=True,
                ax=ax,
                cbar_kws={"label": "Correlation"},
            )
            ax.set_title(f"Class: {class_label}")

        plt.suptitle("Example 2: Progressive Pathway Activation", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig("class_specific_correlations_example2.png", dpi=150, bbox_inches="tight")
        print("✓ Saved Example 2 visualization to class_specific_correlations_example2.png")

    except Exception as e:
        print(f"Note: Visualization skipped ({e})")

    # Save datasets
    print()
    datasets = [
        (x1, y1, meta1, "disease_activated_pathway"),
        (x2, y2, meta2, "progressive_correlation"),
        (X3, y3, meta3, "multiple_pathways"),
    ]

    for i, (x, y, meta, name) in enumerate(datasets, start=1):
        out_path = f"class_specific_example_{i}_{name}.csv"
        to_csv(x, y, meta, out_path)
        print(f"✓ Saved {name} dataset to {out_path}")

    print()
    print("=" * 70)
    print("Key Takeaways:")
    print("  1. Class-specific correlations model biological pathways active only")
    print("     in certain disease states or conditions")
    print("  2. Correlation strength can vary progressively across disease severity")
    print("  3. Multiple pathways can have independent class-specific patterns")
    print("  4. This creates realistic biomarker relationships for ML validation")
    print("=" * 70)


if __name__ == "__main__":
    main()
