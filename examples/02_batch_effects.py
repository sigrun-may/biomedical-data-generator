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

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    effect_type: Literal["additive", "multiplicative"] = "additive",
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
        batch_effects=BatchEffectsConfig(
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
    x1, y1, meta1, cfg1 = create_dataset_with_batches(
        n_batches=3, confounding=0.0, effect_type="additive", effect_strength=0.5
    )
    print(f"Generated dataset: {x1.shape}")
    print("Batch distribution across classes:")
    for class_idx, class_name in enumerate(meta1.class_names):
        batch_counts = np.bincount(meta1.batch_labels[y1 == class_idx])
        print(f"  {class_name}: {batch_counts}")
    print()

    # Example 2: Strong confounding (recruitment bias)
    print("Example 2: Confounded Batch Assignment (Recruitment Bias)")
    print("-" * 70)
    print("Simulating scenario where control samples are mostly from batch 0")
    print("and treated samples are mostly from batch 1.")
    print()
    x2, y2, meta2, cfg2 = create_dataset_with_batches(
        n_batches=2, confounding=0.8, effect_type="additive", effect_strength=0.8
    )
    print(f"Generated dataset: {x2.shape}")
    print("Batch distribution across classes:")
    for class_idx, class_name in enumerate(meta2.class_names):
        batch_counts = np.bincount(meta2.batch_labels[y2 == class_idx])
        print(f"  {class_name}: {batch_counts}")
    print()

    # Example 3: Multiplicative batch effects
    print("Example 3: Multiplicative Batch Effects")
    print("-" * 70)
    print("Simulating instrument calibration differences (scaling factors)")
    print()
    x3, y3, meta3, cfg3 = create_dataset_with_batches(
        n_batches=3, confounding=0.0, effect_type="multiplicative", effect_strength=0.3
    )
    print(f"Generated dataset: {x3.shape}")
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
        feature_idx = 0
        if meta1.batch_labels is not None:
            data1 = {
                "Feature Value": x1.iloc[:, feature_idx],
                "Batch": meta1.batch_labels,
                "Class": y1,
            }

            df1 = pd.DataFrame(data1)
            sns.boxplot(data=df1, x="Batch", y="Feature Value", hue="Class", ax=axes[0])
            axes[0].set_title("No Confounding: Random Batch Assignment")
            axes[0].legend(title="Class")

        # Plot 2: Feature distribution by batch (with confounding)
        if meta2.batch_labels is not None:
            data2 = {
                "Feature Value": x2.iloc[:, feature_idx],
                "Batch": meta2.batch_labels,
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
            (x1, y1, "random_batches"),
            (x2, y2, "confounded_batches"),
            (x3, y3, "multiplicative_batches"),
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
