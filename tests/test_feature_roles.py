# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Unit tests for feature role assignment logic in compute_feature_roles."""


def test_compute_feature_roles_partitions_all_columns():
    """compute_feature_roles assigns every feature to exactly one role."""
    from biomedical_data_generator import (
        ClassConfig,
        CorrClusterConfig,
        DatasetConfig,
        MeanChannel,
        StandaloneInformativeGroup,
        compute_feature_roles,
        generate_dataset,
    )

    cfg = DatasetConfig(
        standalone_informative_groups=[
            StandaloneInformativeGroup(n_features=1, class_sep=1.0)
        ],  # one standalone informative feature
        n_standalone_noise=1,  # one standalone noise feature
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                baseline_correlation=0.7,
                mean_channel=MeanChannel(per_class_effect={1: 1.0}),
            ),
            CorrClusterConfig(n_cluster_features=3, baseline_correlation=0.5),
        ],
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],
        random_state=42,
    )

    _, _, meta = generate_dataset(cfg, return_dataframe=False)
    roles = compute_feature_roles(meta)

    # Matches the index layout asserted in test_generate_dataset_with_noise_anchor_cluster.
    assert roles.standalone_informative_indices == [0]
    assert roles.informative_anchor_indices == [1]
    assert roles.informative_proxy_indices == [2, 3]
    assert roles.noise_anchor_indices == [4]
    assert roles.noise_proxy_indices == [5, 6]
    assert roles.standalone_noise_indices == [7]
    assert roles.cluster_membership == {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1}

    # Every column appears in exactly one role bucket.
    all_role_columns = (
        roles.standalone_informative_indices
        + roles.informative_anchor_indices
        + roles.informative_proxy_indices
        + roles.standalone_noise_indices
        + roles.noise_anchor_indices
        + roles.noise_proxy_indices
    )
    assert sorted(all_role_columns) == list(range(cfg.n_features))
