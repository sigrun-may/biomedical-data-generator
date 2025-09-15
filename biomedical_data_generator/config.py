# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Configuration for dataset generation."""

from pydantic import BaseModel, Field, PositiveInt, NonNegativeInt
from typing import Optional, Tuple, List
import yaml

class DatasetConfig(BaseModel):
    n_samples: PositiveInt = 300
    n_features: PositiveInt = 20
    n_informative: NonNegativeInt = 5
    class_sep: float = 1.2
    weights: Optional[Tuple[float, float]] = None
    random_state: Optional[int] = 42
    # noise / correlations
    n_noise: NonNegativeInt = 0
    noise_dist: str = Field("normal", pattern="^(normal|uniform)$")
    corr_matrix_path: Optional[str] = None
    block_sizes: Optional[List[int]] = None
    corr_within: float = 0.8
    corr_between: float = 0.0

    @classmethod
    def from_yaml(cls, path: str) -> "DatasetConfig":
        with open(path, "r") as f:
            return cls.model_validate(yaml.safe_load(f))
