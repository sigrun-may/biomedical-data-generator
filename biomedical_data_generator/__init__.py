# Copyright (c) 2022 Sigrun May,
# Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2022 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Data generator main package."""

from .config import BatchEffectsConfig, ClassConfig, CorrClusterConfig, DatasetConfig
from .feature_relevance import (
    FeatureRoles,
    RelevanceView,
    compute_feature_roles,
    compute_relevance,
)
from .features.correlated import sample_correlated_data
from .generator import (
    generate_dataset,
)

__all__ = [
    "DatasetConfig",
    "CorrClusterConfig",
    "BatchEffectsConfig",
    "ClassConfig",
    "generate_dataset",
    "sample_correlated_data",
    "FeatureRoles",
    "RelevanceView",
    "compute_feature_roles",
    "compute_relevance",
]
