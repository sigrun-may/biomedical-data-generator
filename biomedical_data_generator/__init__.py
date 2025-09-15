# Copyright (c) 2022 Sigrun May,
# Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2022 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Data generator main package."""

from .core import DatasetConfig, DatasetMeta, generate_dataset
from .config import DatasetConfig
from .core import generate_dataset

def generate_from_yaml(path: str):
    cfg = DatasetConfig.from_yaml(path)
    return generate_dataset(cfg)

__all__ = [
    "generate_dataset",
    "DatasetConfig",
    "DatasetMeta",
]
