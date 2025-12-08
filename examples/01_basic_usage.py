# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Basic usage example for biomedical-data-generator.

This example demonstrates:
- Creating a simple dataset configuration
- Generating synthetic biomedical data
- Understanding the structure of informative, noise, and correlated features
- Exploring the metadata and feature roles
"""

from __future__ import annotations

from biomedical_data_generator.config import ClassConfig, CorrClusterConfig, DatasetConfig
from biomedical_data_generator.generator import generate_dataset
from biomedical_data_generator.utils.export_utils import to_csv


def main() -> None:
    """Generate a basic synthetic biomedical dataset."""
    print("=" * 70)
    print("Basic Usage Example: Synthetic Biomedical Dataset Generation")
    print("=" * 70)
    print()

    # Configure dataset with informative features, noise, and correlated cluster
    cfg = DatasetConfig(
        n_informative=3,
        n_noise=2,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                correlation=0.7,
                structure="equicorrelated",
                anchor_role="informative",
                anchor_effect_size="medium",
                label="Metabolic Pathway",
            )
        ],
        class_configs=[
            ClassConfig(n_samples=100, label="healthy"),
            ClassConfig(n_samples=50, label="diseased"),
        ],
        class_sep=[1.0],  # separation between classes
        random_state=42,
    )

    # Generate dataset
    print("Generating dataset...")
    x, y, meta = generate_dataset(cfg, return_dataframe=True)

    print(f"\n✓ Generated dataset with shape: {x.shape}")
    print(f"  - Samples: {x.shape[0]}")
    print(f"  - Features: {x.shape[1]}")
    print(f"  - Classes: {len(cfg.class_labels)} ({', '.join(cfg.class_labels)})")
    print()

    # Display feature roles
    print("Feature Roles:")
    print(f"  - Informative features: {meta.informative_idx}")
    print(f"  - Noise features: {meta.noise_idx}")
    print(f"  - Cluster features: {list(meta.corr_cluster_indices.values())}")
    for cid, indices in meta.corr_cluster_indices.items():
        anchor = meta.anchor_idx.get(cid)
        # Note: cid is 0-based (cluster 0, 1, 2, ...) but feature names use 1-based (corr1, corr2, ...)
        print(f"    Cluster {cid}: anchor={anchor}, all indices={indices}")
    print()

    # Display sample statistics
    print("Class Distribution:")
    for class_idx, class_name in enumerate(meta.class_names):
        print(f"  - {class_name}: {meta.samples_per_class[class_idx]} samples")

    # Save to CSV
    out_path = "basic_dataset.csv"
    to_csv(x, y, meta, out_path)
    print(f"✓ Saved dataset to {out_path}")
    print()

    # Additional info about feature naming
    print("Feature Naming Convention:")
    print("  - 'i' prefix: informative features")
    print("  - 'n' prefix: noise features")
    print("  - 'corr' prefix: correlated cluster features")
    print("    (corr1_anchor, corr1_2, ... for cluster ID 0)")
    print("    (corr2_anchor, corr2_2, ... for cluster ID 1, etc.)")
    print("    Note: Cluster IDs are 0-based internally, but display names are 1-based")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
