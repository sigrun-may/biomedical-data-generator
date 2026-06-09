# Copyright (c) 2022 Sigrun May,
# Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2022 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Data generator main package."""

from .config import (
    BatchEffectsConfig,
    ClassConfig,
    CorrClusterConfig,
    CovarianceChannel,
    DatasetConfig,
    MeanChannel,
    StandaloneInformativeGroup,
)
from .generator import generate_dataset
from .meta import (
    BatchMeta,
    DatasetMeta,
    FeatureRoles,
    FeatureStrengths,
    compute_feature_roles,
    compute_feature_strengths,
)

__all__ = [
    "DatasetConfig",
    "DatasetMeta",
    "CorrClusterConfig",
    "MeanChannel",
    "CovarianceChannel",
    "StandaloneInformativeGroup",
    "BatchEffectsConfig",
    "ClassConfig",
    "FeatureRoles",
    "FeatureStrengths",
    "BatchMeta",
    "generate_dataset",
    "compute_feature_roles",
    "compute_feature_strengths",
]
