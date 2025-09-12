import numpy as np

from synthetic_data_generator import generate_dataset


def test_shapes_and_types():
    X, y, meta = generate_dataset(
        n_samples=40, n_features=30, n_informative=5, n_noise=20, block_sizes=[5], random_state=0
    )
    assert X.shape == (40, 30)
    assert y.shape == (40,)
    assert isinstance(meta.feature_names, list)
    assert len(meta.feature_names) == 30


def test_reproducibility():
    X1, y1, _ = generate_dataset(random_state=123)
    X2, y2, _ = generate_dataset(random_state=123)
    assert np.allclose(X1, X2)
    assert np.array_equal(y1, y2)


def test_effect_size_direction():
    X, y, meta = generate_dataset(
        n_samples=200, n_features=20, n_informative=5, class_sep=1.0, n_noise=10, block_sizes=[5, 5], random_state=7
    )
    # mean difference over informative features ~ positive
    mu0 = X[y == 0][:, :5].mean()
    mu1 = X[y == 1][:, :5].mean()
    assert (mu1 - mu0) > 0
