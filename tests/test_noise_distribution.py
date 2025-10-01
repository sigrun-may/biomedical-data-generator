import numpy as np
import pytest

from biomedical_data_generator import DatasetConfig, NoiseDistribution
from biomedical_data_generator.generator import generate_dataset


def _extract_noise_block(X, meta):
    # meta.noise_idx is expected per the generator's DatasetMeta
    noise_idx = meta.noise_idx
    assert isinstance(noise_idx, list)
    Xn = X[:, noise_idx] if not hasattr(X, "iloc") else X.iloc[:, noise_idx].to_numpy()
    return Xn

@pytest.mark.parametrize("dist", [NoiseDistribution.normal, NoiseDistribution.uniform, NoiseDistribution.laplace])
def test_noise_distributions_basic(dist):
    cfg = DatasetConfig(
        n_samples=120, n_informative=2, n_pseudo=1, n_noise=4,
        noise_distribution=dist, noise_scale=1.5,
        n_features=2 + 1 + 4,
        random_state=42,
    )
    X, y, meta = generate_dataset(cfg, return_dataframe=False)
    assert X.shape == (120, cfg.n_features) and y.shape == (120,)
    Xn = _extract_noise_block(X, meta)
    assert Xn.shape == (120, 4)

    # sanity by distribution family
    if dist == NoiseDistribution.uniform:
        # with scale=1.5 and default low/high = ±scale, we expect values in [-1.5, 1.5] approx.
        assert Xn.min() >= -3.0 and Xn.max() <= 3.0
    else:
        # heavy tails not too extreme with scale=1.5; check std is in a reasonable band
        std = float(np.std(Xn))
        assert 0.5 <= std <= 3.0

def test_noise_params_override_scale():
    # scale would be 2.0 by noise_scale, but we override to 0.3 via noise_params['scale']
    cfg = DatasetConfig(
        n_samples=100, n_informative=1, n_pseudo=0, n_noise=2,
        noise_distribution=NoiseDistribution.normal,
        noise_scale=2.0,
        noise_params={"scale": 0.3, "loc": 0.0},
        n_features=1 + 0 + 2,
        random_state=0,
    )
    X, y, meta = generate_dataset(cfg, return_dataframe=False)
    Xn = _extract_noise_block(X, meta)
    std = float(np.std(Xn))
    assert std < 1.0  # should reflect the overridden smaller scale

def test_determinism_random_state():
    cfg = DatasetConfig(
        n_samples=80, n_informative=2, n_pseudo=1, n_noise=3,
        noise_distribution=NoiseDistribution.laplace,
        noise_scale=1.0,
        n_features=2 + 1 + 3,
        random_state=7,
    )
    X1, y1, meta1 = generate_dataset(cfg, return_dataframe=False)
    X2, y2, meta2 = generate_dataset(cfg, return_dataframe=False)
    assert np.allclose(X1, X2)
    assert np.array_equal(y1, y2)
    assert meta1.y_counts == meta2.y_counts
