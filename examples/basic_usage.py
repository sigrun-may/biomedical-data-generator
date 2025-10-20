# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Basic usage example for biomedical_data_generator."""
from __future__ import annotations

import pandas as pd

from biomedical_data_generator import CorrCluster, DatasetConfig, generate_dataset


def main() -> None:
    """Generate a small dataset and preview its correlation matrix."""
    cfg = DatasetConfig(
        n_samples=200,
        n_features=10,  # = informative + pseudo + noise + proxies_from_clusters
        n_informative=4,
        n_pseudo=2,
        n_noise=4,
        n_classes=2,
        weights=[0.6, 0.4],
        effect_size="medium",
        corr_between=0.1,
        corr_clusters=[
            CorrCluster(
                n_cluster_features=3,
                rho=0.7,
                structure="equicorrelated",
                anchor_role="informative",
                anchor_beta=1.0,
            )
        ],
        random_state=42,
    )

    X, y, meta = generate_dataset(cfg)
    # Ensure we have a DataFrame for correlation preview
    df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    corr = df.corr(numeric_only=True)
    print(f"X shape: {df.shape} | y length: {len(y)}")
    print("Top-left of correlation matrix:")
    print(corr.round(2).iloc[:5, :5])

    out_path = "examples/basic_dataset.csv"
    df_out = df.copy()
    df_out["target"] = y  # add labels
    df_out.to_csv(out_path, index=False)
    print(f"Saved dataset to {out_path}")


if __name__ == "__main__":
    main()
