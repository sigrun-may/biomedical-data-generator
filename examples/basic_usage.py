# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Basic usage example for biomedical-data-generator."""

from __future__ import annotations

from biomedical_data_generator.config import ClassConfig, CorrClusterConfig, DatasetConfig
from biomedical_data_generator.generator import generate_dataset


def main() -> None:
    """Basic usage example for biomedical-data-generator."""
    # Simple config: 2 free informative features, 2 noise features, one correlated cluster
    cfg = DatasetConfig(
        n_informative=3,
        n_noise=2,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                rho=0.7,
                structure="equicorrelated",
                anchor_role="informative",
                anchor_effect_size="medium",
            )
        ],
        class_configs=[
            ClassConfig(n_samples=100),  # label → "class_0"
            ClassConfig(n_samples=50),  # label → "class_1"
        ],  # two classes
        class_sep=[1.0],  # separation between class_0 and class_1
        random_state=42,
    )

    # generate_dataset returns (DataFrame, y, meta) by default
    X, y, meta = generate_dataset(cfg)

    print(f"X shape: {X.shape}, y length: {len(y)}")
    print("First 5 rows:")
    print(X.head())
    print("\nMeta summary:")

    # Save to CSV
    out_path = "basic_dataset.csv"
    df_out = X.copy()
    df_out["target"] = y
    df_out.to_csv(out_path, index=False)
    print(f"Saved dataset to {out_path}")


if __name__ == "__main__":
    main()
