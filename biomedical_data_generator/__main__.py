# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Command line interface for dataset generation."""

from __future__ import annotations
import argparse, json
import pandas as pd
from .config import DatasetConfig
from .generator import generate_dataset

def main():
    ap = argparse.ArgumentParser("biomedical-data-generator")
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--out", default=None, help="CSV output (optional)")
    args = ap.parse_args()

    cfg = DatasetConfig.from_yaml(args.config)
    X, y, meta = generate_dataset(cfg)

    if args.out:
        df = pd.DataFrame(X, columns=meta.feature_names)
        df["class"] = y
        df.to_csv(args.out, index=False)

    print(json.dumps(meta.to_dict(), indent=2))

if __name__ == "__main__":
    main()

