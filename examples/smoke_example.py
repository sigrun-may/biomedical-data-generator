import numpy as np

from biomedical_data_generator import CorrCluster, DatasetConfig, NoiseDistribution
from biomedical_data_generator.generator import generate_correlated_cluster, generate_dataset

# 1) Toeplitz-Cluster
Xc, meta_c = generate_correlated_cluster(200, 5, rho=0.6, structure="toeplitz", random_state=0)
C = np.corrcoef(Xc, rowvar=False)
assert C.shape == (5, 5) and np.allclose(np.diag(C), 1, atol=1e-6)

# 2) Noise-Verteilungen
cfg = DatasetConfig(
    n_samples=150,
    n_informative=2,
    n_pseudo=1,
    n_noise=3,
    noise_distribution=NoiseDistribution.uniform,
    noise_scale=2.0,
    n_features=2 + 1 + 3,  # = required
    random_state=42,
)
X, y, meta = generate_dataset(cfg, return_dataframe=False)
assert X.shape == (150, cfg.n_features) and y.shape == (150,)
# Uniform sollte grob in [-2, 2] liegen:
noise_block = X[:, meta.noise_idx]
assert np.max(noise_block) <= 3 and np.min(noise_block) >= -3

# 3) equicorrelated/weights funktionieren
cfg2 = DatasetConfig(
    n_samples=300,
    n_informative=2,
    n_pseudo=0,
    n_noise=0,
    corr_clusters=[
        CorrCluster(size=4, rho=0.7, structure="equicorrelated", anchor_role="informative", anchor_beta=1.0)
    ],
    n_features=2 + (4 - 1),  # informative + proxies
    n_classes=3,
    weights=[0.2, 0.5, 0.3],
    random_state=1,
)
X2, y2, meta2 = generate_dataset(cfg2, return_dataframe=False)
assert X2.shape == (300, cfg2.n_features) and set(np.unique(y2)) <= {0, 1, 2}
