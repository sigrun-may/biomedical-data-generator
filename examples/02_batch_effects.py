# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Batch effects example for biomedical-data-generator.

This example demonstrates:
- Simulating batch effects (site-to-site differences, instrument variations)
- Confounding batch effects with class labels (recruitment bias)
- Comparing additive vs multiplicative batch effects
- Visualizing the impact of batch effects on feature distributions
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from biomedical_data_generator.config import (
    BatchEffectsConfig,
    ClassConfig,
    CorrClusterConfig,
    DatasetConfig,
)
from biomedical_data_generator.generator import generate_dataset


def create_dataset_with_batches(
    n_batches: int,
    confounding: float,
    effect_type: str = "additive",
    effect_strength: float = 0.5,
) -> tuple:
    """Create a dataset with specified batch effect configuration."""
    cfg = DatasetConfig(
        n_informative=5,
        n_noise=3,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=4,
                correlation=0.8,
                structure="equicorrelated",
                anchor_role="informative",
                anchor_effect_size="medium",
                label="Immune Response",
            )
        ],
        class_configs=[
            ClassConfig(n_samples=100, label="control"),
            ClassConfig(n_samples=100, label="treated"),
        ],
        class_sep=[1.5],
        batch=BatchEffectsConfig(
            n_batches=n_batches,
            effect_strength=effect_strength,
            effect_type=effect_type,
            confounding_with_class=confounding,
            affected_features="all",
        ),
        random_state=42,
    )

    X, y, meta = generate_dataset(cfg)
    return X, y, meta, cfg


def main() -> None:
    """Demonstrate batch effects in synthetic biomedical datasets."""
    print("=" * 70)
    print("Batch Effects Example: Simulating Multi-Center Study Artifacts")
    print("=" * 70)
    print()

    # Example 1: No confounding (random batch assignment)
    print("Example 1: Random Batch Assignment (No Confounding)")
    print("-" * 70)
    X1, y1, meta1, cfg1 = create_dataset_with_batches(
        n_batches=3, confounding=0.0, effect_type="additive", effect_strength=0.5
    )
    print(f"Generated dataset: {X1.shape}")
    print(f"Batch distribution across classes:")
    batch_col = meta1.batch_labels if hasattr(meta1, "batch_labels") else None
    if batch_col is not None:
        for class_label in ["control", "treated"]:
            mask = y1 == class_label
            batch_counts = np.bincount(batch_col[mask])
            print(f"  {class_label}: {batch_counts}")
    print()

    # Example 2: Strong confounding (recruitment bias)
    print("Example 2: Confounded Batch Assignment (Recruitment Bias)")
    print("-" * 70)
    print("Simulating scenario where control samples are mostly from batch 0")
    print("and treated samples are mostly from batch 1.")
    print()
    X2, y2, meta2, cfg2 = create_dataset_with_batches(
        n_batches=2, confounding=0.8, effect_type="additive", effect_strength=0.8
    )
    print(f"Generated dataset: {X2.shape}")
    print(f"Batch distribution across classes:")
    batch_col2 = meta2.batch_labels if hasattr(meta2, "batch_labels") else None
    if batch_col2 is not None:
        for class_label in ["control", "treated"]:
            mask = y2 == class_label
            batch_counts = np.bincount(batch_col2[mask])
            print(f"  {class_label}: {batch_counts}")
    print()

    # Example 3: Multiplicative batch effects
    print("Example 3: Multiplicative Batch Effects")
    print("-" * 70)
    print("Simulating instrument calibration differences (scaling factors)")
    print()
    X3, y3, meta3, cfg3 = create_dataset_with_batches(
        n_batches=3, confounding=0.0, effect_type="multiplicative", effect_strength=0.3
    )
    print(f"Generated dataset: {X3.shape}")
    print()

    # Example 4: Affecting only informative features
    print("Example 4: Batch Effects on Informative Features Only")
    print("-" * 70)
    cfg4 = DatasetConfig(
        n_informative=5,
        n_noise=5,
        class_configs=[
            ClassConfig(n_samples=100, label="control"),
            ClassConfig(n_samples=100, label="treated"),
        ],
        class_sep=[1.5],
        batch=BatchEffectsConfig(
            n_batches=3,
            effect_strength=0.7,
            effect_type="additive",
            confounding_with_class=0.0,
            affected_features="informative",  # Only affect informative features
        ),
        random_state=42,
    )
    X4, y4, meta4 = generate_dataset(cfg4)
    print(f"Generated dataset: {X4.shape}")
    print(f"Batch effects applied to: {cfg4.batch.affected_features} features")
    print(f"  - Informative feature indices: {meta4.informative_idx}")
    print(f"  - Noise feature indices: {meta4.noise_idx}")
    print()

    # Visualization (optional, requires matplotlib)
    print("Visualization Tips:")
    print("-" * 70)
    print("To visualize batch effects:")
    print("1. Create box plots or violin plots grouped by batch")
    print("2. Use PCA and color points by batch vs. class")
    print("3. Calculate within-batch and between-batch variance ratios")
    print("4. Apply batch correction methods (e.g., ComBat) and compare")
    print()

    # Simple visualization example
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Feature distribution by batch (no confounding)
        if batch_col is not None:
            feature_idx = 0
            data1 = {
                "Feature Value": X1.iloc[:, feature_idx],
                "Batch": batch_col,
                "Class": y1,
            }
            import pandas as pd

            df1 = pd.DataFrame(data1)
            sns.boxplot(data=df1, x="Batch", y="Feature Value", hue="Class", ax=axes[0])
            axes[0].set_title("No Confounding: Random Batch Assignment")
            axes[0].legend(title="Class")

        # Plot 2: Feature distribution by batch (with confounding)
        if batch_col2 is not None:
            data2 = {
                "Feature Value": X2.iloc[:, feature_idx],
                "Batch": batch_col2,
                "Class": y2,
            }
            df2 = pd.DataFrame(data2)
            sns.boxplot(data=df2, x="Batch", y="Feature Value", hue="Class", ax=axes[1])
            axes[1].set_title("Strong Confounding: Recruitment Bias")
            axes[1].legend(title="Class")

        plt.tight_layout()
        plt.savefig("batch_effects_comparison.png", dpi=150)
        print("✓ Saved visualization to batch_effects_comparison.png")
        print()
    except Exception as e:
        print(f"Note: Visualization skipped ({e})")
        print()

    # Save datasets
    for i, (X, y, name) in enumerate(
        [
            (X1, y1, "random_batches"),
            (X2, y2, "confounded_batches"),
            (X3, y3, "multiplicative_batches"),
            (X4, y4, "informative_only_batches"),
        ],
        start=1,
    ):
        out_path = f"batch_example_{i}_{name}.csv"
        df_out = X.copy()
        df_out["target"] = y
        df_out.to_csv(out_path, index=False)
        print(f"✓ Saved {name} dataset to {out_path}")

    print()
    print("=" * 70)
    print("Key Takeaways:")
    print("  1. Batch effects simulate technical variation in multi-center studies")
    print("  2. Confounding creates recruitment bias (class-batch correlation)")
    print("  3. Additive effects shift feature means; multiplicative effects scale variance")
    print("  4. Batch effects can be applied to all features or specific subsets")
    print("=" * 70)


if __name__ == "__main__":
    main()
