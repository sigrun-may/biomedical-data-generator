# Copyright (c) 2022 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import numpy as np

import synthetic_data_generator as sdg


def test_public_entrypoint_exists():
    # Top-level import should expose the stable entry points
    assert hasattr(sdg, "generate_dataset")
    assert hasattr(sdg, "DatasetMeta")

def test_generate_dataset_smoke():
    # Basic smoke test ensuring shapes and return-meta behavior
    X, y, meta = sdg.generate_dataset(
        n_samples=12, n_features=20, n_informative=5, random_state=0
    )
    assert X.shape == (12, X.shape[1])
    assert y.shape == (12,)
    assert len(meta.feature_names) == X.shape[1]

def test_return_meta_flag():
    X, y = sdg.generate_dataset(n_samples=10, n_features=15, return_meta=False)
    assert X.shape == (10, 15)
    assert y.shape == (10,)


def test_shapes_and_types():
    X, y, meta = sdg.generate_dataset(
        n_samples=40, n_features=30, n_informative=5, n_noise=20, block_sizes=[5], random_state=0
    )
    assert X.shape == (40, 30)
    assert y.shape == (40,)
    assert isinstance(meta.feature_names, list)
    assert len(meta.feature_names) == 30


def test_reproducibility():
    X1, y1, _ = sdg.generate_dataset(random_state=123)
    X2, y2, _ = sdg.generate_dataset(random_state=123)
    assert np.allclose(X1, X2)
    assert np.array_equal(y1, y2)


def test_effect_size_direction():
    X, y, meta = sdg.generate_dataset(
        n_samples=200, n_features=20, n_informative=5, class_sep=1.0, n_noise=10, block_sizes=[5, 5], random_state=7
    )
    # mean difference over informative features ~ positive
    mu0 = X[y == 0][:, :5].mean()
    mu1 = X[y == 1][:, :5].mean()
    assert (mu1 - mu0) > 0
