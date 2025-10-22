# # Copyright (c) 2022 Sigrun May,
# # Ostfalia Hochschule für angewandte Wissenschaften
# #
# # This software is distributed under the terms of the MIT license
# # which is available at https://opensource.org/licenses/MIT
#
# """Tests for the multiclass synthetic dataset generator."""
#
# from __future__ import annotations
#
# import numpy as np
# import pandas as pd
# import pytest
#
# from biomedical_data_generator.config import CorrCluster, DatasetConfig
# from biomedical_data_generator.features.correlated import find_seed_for_correlation, sample_cluster
# from biomedical_data_generator.generator import (
#     DatasetMeta,
#     find_dataset_seed_for_class_weights,
#     find_dataset_seed_for_score,
#     generate_dataset,
# )
#
#
# def _class_means_for_column(X, y, col_name: str):
#     """Helper: compute per-class means of a named column (works with DataFrame)."""
#     assert isinstance(X, pd.DataFrame)
#     s = X[col_name]
#     means = []
#     for k in sorted(np.unique(y)):
#         means.append(float(s[y == k].mean()))
#     return means
#
#
# def test_default_returns_dataframe_and_meta_types():
#     cfg = DatasetConfig(
#         n_samples=200,
#         n_features=10,
#         n_informative=3,
#         n_pseudo=3,
#         n_noise=2,
#         n_classes=3,
#         random_state=42,
#         corr_clusters=[
#             CorrCluster(
#                 n_cluster_features=3, rho=0.6, anchor_role="informative", anchor_effect_size=1.0, anchor_class=2
#             )
#         ],
#     )
#     X, y, meta = generate_dataset(cfg)  # default: DataFrame
#
#     assert isinstance(X, pd.DataFrame)
#     assert isinstance(y, np.ndarray) and y.dtype.kind in ("i", "u")
#     assert isinstance(meta, DatasetMeta)
#
#     # Shapes
#     assert X.shape == (cfg.n_samples, cfg.n_features)
#     assert y.shape == (cfg.n_samples,)
#     # Column names align with meta
#     assert list(X.columns) == meta.feature_names
#
#
# def test_ndarray_fallback_and_reproducibility():
#     cfg = DatasetConfig(
#         n_samples=150,
#         n_features=6,
#         n_informative=2,
#         n_pseudo=2,
#         n_noise=2,
#         n_classes=2,
#         random_state=7,
#     )
#     X1, y1, m1 = generate_dataset(cfg, return_dataframe=False)
#     X2, y2, m2 = generate_dataset(cfg, return_dataframe=False)
#
#     assert isinstance(X1, np.ndarray) and isinstance(X2, np.ndarray)
#     assert X1.shape == X2.shape == (cfg.n_samples, cfg.n_features)
#     assert np.array_equal(X1, X2)
#     assert np.array_equal(y1, y2)
#     # meta equality: compare a few robust fields
#     assert m1.feature_names == m2.feature_names
#     assert m1.y_counts == m2.y_counts
#     assert m1.y_weights == m2.y_weights
#
#
# def test_feature_order_and_prefixes_prefixed_naming():
#     cfg = DatasetConfig(
#         n_samples=100,
#         n_features=9,  # <-- 3 (cluster) + 2 (free i) + 2 (free p) + 2 (noise)
#         n_informative=3,
#         n_pseudo=2,  # only free pseudos
#         n_noise=2,
#         n_classes=2,
#         random_state=0,
#         feature_naming="prefixed",
#         corr_clusters=[
#             CorrCluster(
#                 n_cluster_features=3, rho=0.5, anchor_role="informative", anchor_effect_size=1.0, anchor_class=1
#             )
#         ],
#     )
#     X, y, meta = generate_dataset(cfg)
#     cols = list(X.columns)
#
#     # Expected order: clusters (anchor first, then corr proxies) -> free informative -> free pseudo -> noise
#     # With size=3 cluster: first = i1 (anchor), then corr1_2, corr1_3
#     assert cols[0].startswith("i")
#     assert cols[1].startswith("corr1_") and cols[2].startswith("corr1_")
#
#     # Then free informative (i2, i3)
#     assert cols[3].startswith("i") and cols[4].startswith("i")
#
#     # Then pseudo (p*) then noise (n*)
#     assert cols[5].startswith("p")
#     assert cols[6].startswith("p")
#     assert cols[7].startswith("n")
#
#
# def test_multiclass_anchor_influences_its_class_mean():
#     # One informative cluster (size=4) with anchor boosting class 2
#     cfg = DatasetConfig(
#         n_samples=1200,
#         n_features=13,
#         n_informative=4,  # 1 anchor + 3 free informative
#         n_pseudo=3,
#         n_noise=3,
#         n_classes=3,
#         class_sep=1.5,
#         random_state=11,
#         corr_clusters=[
#             CorrCluster(
#                 n_cluster_features=4, rho=0.6, anchor_role="informative", anchor_effect_size=1.2, anchor_class=2
#             )
#         ],
#     )
#     X, y, meta = generate_dataset(cfg)
#
#     # Anchor column is the very first column (i1) per naming/ordering rules
#     anchor_col = meta.feature_names[0]
#     means = _class_means_for_column(X, y, anchor_col)
#     # We expect the anchor feature to have the largest mean in its target class (index 2)
#     # Use a tolerant inequality to reduce flaky risk.
#     assert means[2] > means[0] - 1e-6
#     assert means[2] > means[1] - 1e-6
#
#
# def test_class_weights_bias_matches_priors_approximately():
#     # Aim for skewed priors; empirical props should be close.
#     weights = [0.7, 0.2, 0.1]
#     cfg = DatasetConfig(
#         n_samples=3000,  # larger n reduces sampling noise
#         n_features=14,
#         n_informative=4,
#         n_pseudo=4,
#         n_noise=4,
#         n_classes=3,
#         weights=weights,
#         class_sep=1.0,
#         random_state=123,
#         corr_clusters=[
#             CorrCluster(
#                 n_cluster_features=3, rho=0.5, anchor_role="informative", anchor_effect_size=1.0, anchor_class=0
#             )
#         ],
#     )
#     X, y, meta = generate_dataset(cfg)
#
#     target = np.array(weights, dtype=float) / np.sum(weights)
#     empirical = np.array([meta.y_weights[k] for k in range(cfg.n_classes)], dtype=float)
#     l1 = float(np.abs(empirical - target).sum())
#     assert l1 <= 0.08  # reasonably tight with n=3000
#
#
# def test_informative_separation():
#     cfg = DatasetConfig(
#         n_samples=200,
#         n_informative=2,
#         n_pseudo=0,
#         n_noise=0,
#         n_features=2,
#         n_classes=2,
#         class_sep=10,
#         random_state=42,
#     )
#     X, y, meta = generate_dataset(cfg)
#     for idx in meta.informative_idx:
#         mean0 = X.iloc[y == 0, idx].mean()
#         mean1 = X.iloc[y == 1, idx].mean()
#         print(f"Feature {idx}: mean0={mean0}, mean1={mean1}, separation={abs(mean0 - mean1)}")
#         assert abs(mean0 - mean1) > 5  # separation threshold
#
#
# def test_output_shapes():
#     cfg = DatasetConfig(
#         n_samples=100,
#         n_informative=1,
#         n_pseudo=1,
#         n_noise=1,
#         n_features=3,
#         n_classes=2,
#     )
#     X, y, meta = generate_dataset(cfg)
#     assert X.shape == (cfg.n_samples, cfg.n_features)
#     assert y.shape == (cfg.n_samples,)
#
#
# def test_generate_correlated_cluster_and_find_seed_for_correlation():
#     n = 400
#     size = 5
#     rho_target = 0.65
#     seed, meta = find_seed_for_correlation(
#         n_samples=n,
#         n_cluster_features=size,
#         rho_target=rho_target,
#         structure="equicorrelated",
#         tol=0.03,
#         start_seed=0,
#         max_tries=200,
#     )
#     assert isinstance(seed, int)
#     assert "mean_offdiag" in meta and "min_offdiag" in meta
#     assert abs(meta["mean_offdiag"] - rho_target) <= 0.07  # acceptance tol is 0.03, allow a bit more slack here
#
#
# def test_dataset_seed_for_class_weights_helper():
#     cfg = DatasetConfig(
#         n_samples=800,
#         n_features=11,
#         n_informative=3,
#         n_pseudo=3,
#         n_noise=3,
#         n_classes=3,
#         weights=[0.5, 0.3, 0.2],
#         random_state=None,  # will be set by the helper
#         corr_clusters=[
#             CorrCluster(
#                 n_cluster_features=3, rho=0.6, anchor_role="informative", anchor_effect_size=1.0, anchor_class=1
#             )
#         ],
#     )
#     seed, X, y, meta = find_dataset_seed_for_class_weights(cfg, tol=0.06, start_seed=0, max_tries=200)
#     assert isinstance(seed, int)
#     # Check proportions
#     emp = np.array([meta.y_weights[k] for k in range(cfg.n_classes)], dtype=float)
#     tgt = np.array(cfg.weights, dtype=float) / np.sum(cfg.weights)
#     assert float(np.abs(emp - tgt).sum()) <= 0.06
#
#
# def test_dataset_seed_for_score_helper_max_mode():
#     # Simple separability proxy: average absolute correlation of informative cols vs. a one-vs-rest y signal
#     def sep_score(X, y, meta) -> float:
#         if isinstance(X, pd.DataFrame):
#             Z = X.to_numpy()
#         else:
#             Z = X
#         # Build a simple continuous target: class index standardized
#         y_cont = (y - y.mean()) / (y.std() + 1e-9)
#         cols = meta.informative_idx
#         if not cols:
#             return 0.0
#         corr = np.corrcoef(Z[:, cols].T, y_cont)[-1, :-1]
#         return float(np.nanmean(np.abs(corr)))
#
#     cfg = DatasetConfig(
#         n_samples=600,
#         n_features=12,
#         n_informative=4,
#         n_pseudo=3,
#         n_noise=3,
#         n_classes=3,
#         class_sep=1.0,
#         random_state=None,
#         corr_clusters=[
#             CorrCluster(
#                 n_cluster_features=3, rho=0.5, anchor_role="informative", anchor_effect_size=1.0, anchor_class=0
#             )
#         ],
#     )
#     seed, X, y, meta, score = find_dataset_seed_for_score(
#         cfg, sep_score, mode="max", threshold=0.12, start_seed=0, max_tries=150
#     )
#     assert isinstance(seed, int)
#     assert isinstance(score, float)
#     assert score >= 0.12
#
#
# def test_validation_errors_and_edge_cases():
#     # n_classes must be >= 2
#     with pytest.raises(ValueError):
#         _ = generate_dataset(
#             DatasetConfig(
#                 n_samples=50,
#                 n_features=4,
#                 n_informative=2,
#                 n_pseudo=1,
#                 n_noise=1,
#                 n_classes=1,
#             )
#         )
#
#     # weights length mismatch
#     with pytest.raises(ValueError):
#         _ = generate_dataset(
#             DatasetConfig(
#                 n_samples=50,
#                 n_features=5,
#                 n_informative=2,
#                 n_pseudo=2,
#                 n_noise=1,
#                 n_classes=3,
#                 weights=[0.5, 0.5],  # wrong length
#             )
#         )
#
#     # correlated cluster parameter validation in cluster-only generator
#     with pytest.raises(ValueError):
#         _ = sample_cluster(n_samples=100, n_features=0, rho=0.5)
#
#     with pytest.raises(ValueError):
#         _ = sample_cluster(n_samples=100, n_features=3, rho=1.1)
