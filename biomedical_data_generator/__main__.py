# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Command line interface for dataset generation."""

import argparse, json
from . import generate_from_yaml
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--out", help="CSV path")
args = parser.parse_args()
X, y, meta = generate_from_yaml(args.config)
if args.out:
    import pandas as pd
    df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(X.shape[1])])
    df["class"] = y
    df.to_csv(args.out, index=False)
print(json.dumps(meta, indent=2))
