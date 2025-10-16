# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tests for batch effects."""

# import numpy as np
# from biomedical_data_generator.config import DatasetConfig, BatchConfig
# from biomedical_data_generator.generator import generate_dataset
#
# def test_batch_effects_shift_means():
#     cfg = DatasetConfig(
#         n_samples=200,
#         n_informative=3,
#         n_noise=5,
#         n_features=8,  # = required
#         random_state=42,
#         batch=BatchConfig(n_batches=4, sigma=0.8, confounding=False),
#     )
#     X, y, meta = generate_dataset(cfg, return_dataframe=False)
#     re = meta.batch
#     col = meta.informative_idx[0]
#     means = [X[re.batch_ids == g, col].mean() for g in range(len(re.batch_offsets))]
#     assert np.std(means) > 0.1
