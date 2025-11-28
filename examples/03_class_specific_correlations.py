# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Class-specific correlations example for biomedical-data-generator.

This example demonstrates:
- Creating feature clusters with class-specific correlation patterns
- Simulating pathway activation that differs between disease states
- Modeling biomarker relationships that only exist in certain conditions
- Comparing correlation matrices across classes
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from biomedical_data_generator.config import ClassConfig, CorrClusterConfig, DatasetConfig
from biomedical_data_generator.generator import generate_dataset


def compute_correlation_by_class(X: pd.DataFrame, y: pd.Series, feature_indices: list[int]) -> dict:
    """Compute correlation matrices for specified features, grouped by class."""
    correlations = {}
    for class_label in np.unique(y):
        mask = y == class_label
        X_class = X[mask].iloc[:, feature_indices]
        correlations[class_label] = X_class.corr()
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
        n_informative=3,
        n_noise=2,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=4,
                correlation={0: 0.0, 1: 0.85},  # No correlation in class 0, strong in class 1
                structure="equicorrelated",
                anchor_role="informative",
                anchor_effect_size="large",
                anchor_class=1,
                label="Disease-Activated Immune Pathway",
            )
        ],
        class_configs=[
            ClassConfig(n_samples=100, label="healthy"),
            ClassConfig(n_samples=100, label="diseased"),
        ],
        class_sep=[1.5],
        random_state=42,
    )
    X1, y1, meta1 = generate_dataset(cfg1)
    print(f"✓ Generated dataset: {X1.shape}")
    print()

    # Compute correlations per class
    cluster_id = 0  # Cluster IDs are 0-based (0, 1, 2, ...)
    cluster_features = meta1.corr_cluster_indices[cluster_id]
    correlations1 = compute_correlation_by_class(X1, y1, cluster_features)

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
        n_informative=4,
        n_noise=3,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=5,
                correlation={0: 0.3, 1: 0.7, 2: 0.9},  # Increasing correlation strength
                structure="equicorrelated",
                anchor_role="informative",
                anchor_effect_size="medium",
                label="Metabolic Dysregulation Pathway",
            )
        ],
        class_configs=[
            ClassConfig(n_samples=80, label="healthy"),
            ClassConfig(n_samples=80, label="mild_disease"),
            ClassConfig(n_samples=80, label="severe_disease"),
        ],
        class_sep=[1.0, 1.5],  # Separation between consecutive classes
        random_state=42,
    )

    X2, y2, meta2 = generate_dataset(cfg2)
    print(f"✓ Generated dataset: {X2.shape}")
    print()

    # Compute correlations per class
    cluster_id = 0  # First cluster (0-based indexing)
    cluster_features2 = meta2.corr_cluster_indices[cluster_id]
    correlations2 = compute_correlation_by_class(X2, y2, cluster_features2)

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
        n_informative=6,
        n_noise=4,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                correlation={0: 0.8, 1: 0.0},  # Pathway A: active in class 0
                structure="equicorrelated",
                anchor_role="informative",
                anchor_effect_size="medium",
                anchor_class=0,
                label="Pathway A (Class 0 Specific)",
            ),
            CorrClusterConfig(
                n_cluster_features=3,
                correlation={0: 0.0, 1: 0.8},  # Pathway B: active in class 1
                structure="equicorrelated",
                anchor_role="informative",
                anchor_effect_size="medium",
                anchor_class=1,
                label="Pathway B (Class 1 Specific)",
            ),
        ],
        class_configs=[
            ClassConfig(n_samples=100, label="subtype_A"),
            ClassConfig(n_samples=100, label="subtype_B"),
        ],
        class_sep=[2.0],
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
        (X1, y1, "disease_activated_pathway"),
        (X2, y2, "progressive_correlation"),
        (X3, y3, "multiple_pathways"),
    ]

    for i, (X, y, name) in enumerate(datasets, start=1):
        out_path = f"class_specific_example_{i}_{name}.csv"
        df_out = X.copy()
        df_out["target"] = y
        df_out.to_csv(out_path, index=False)
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
