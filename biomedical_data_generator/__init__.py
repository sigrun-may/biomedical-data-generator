# Copyright (c) 2022 Sigrun May,
# Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2022 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Data generator main package."""

from .config import BatchEffectsConfig, ClassConfig, CorrClusterConfig, DatasetConfig
from .features.correlated import sample_correlated_data
from .generator import (
    generate_dataset,
)
from .utils.correlation_tools import find_seed_for_correlation

__all__ = [
    "DatasetConfig",
    "CorrClusterConfig",
    "BatchEffectsConfig",
    "ClassConfig",
    "generate_dataset",
    "sample_correlated_data",
    "find_seed_for_correlation",
]
