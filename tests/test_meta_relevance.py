# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tests for signal-derived feature relevance in :mod:`biomedical_data_generator.meta`.

The private predicate ``_cluster_is_informative`` derives a cluster's relevance
from the generated signal: a cluster is informative when its mean channel varies
across classes (first moment) or its within-cluster correlation varies across
classes (second moment / differential co-expression). These tests pin the
predicate on primitives and verify the resulting ``FeatureRoles`` partition for
a covariance-only noise anchor. They assert on indices / roles / partition only.
"""

import pytest
from pydantic import ValidationError

from biomedical_data_generator import (
    ClassConfig,
    CorrClusterConfig,
    CovarianceChannel,
    DatasetConfig,
    MeanChannel,
    compute_feature_roles,
    compute_feature_strengths,
    generate_dataset,
)
from biomedical_data_generator.config import StandaloneInformativeGroup
from biomedical_data_generator.meta import _cluster_is_informative, _proxy_mean_offset

_TOL = 1e-9


# ---------------------------------------------------------------------------
# Predicate unit tests (primitives only; no config / validator interaction)
# ---------------------------------------------------------------------------
def test_predicate_covariance_only_noise_anchor_is_informative():
    """A class-varying correlation (no mean channel) is informative."""
    assert (
        _cluster_is_informative(
            mean_per_class=None,
            covariance_per_class={0: 0.0, 1: 0.8},
            baseline_correlation=0.0,
            n_classes=2,
        )
        is True
    )


def test_predicate_class_uniform_mean_is_noise():
    """A mean channel that is equal across classes carries no signal."""
    assert (
        _cluster_is_informative(
            mean_per_class={0: 1.0, 1: 1.0},
            covariance_per_class=None,
            baseline_correlation=0.5,
            n_classes=2,
        )
        is False
    )


def test_predicate_genuine_mean_is_informative():
    """A mean channel that differs across classes is discriminative."""
    assert (
        _cluster_is_informative(
            mean_per_class={1: 1.0},
            covariance_per_class=None,
            baseline_correlation=0.5,
            n_classes=2,
        )
        is True
    )


def test_predicate_genuine_noise_is_noise():
    """No channels and a constant baseline correlation carry no signal."""
    assert (
        _cluster_is_informative(
            mean_per_class=None,
            covariance_per_class=None,
            baseline_correlation=0.5,
            n_classes=2,
        )
        is False
    )


def test_predicate_constant_correlation_mapping_is_noise():
    """A covariance mapping with equal values does not vary across classes."""
    assert (
        _cluster_is_informative(
            mean_per_class=None,
            covariance_per_class={0: 0.5, 1: 0.5},
            baseline_correlation=0.5,
            n_classes=2,
        )
        is False
    )


def test_predicate_both_channels_is_informative():
    """A varying mean channel and a varying covariance channel are both signal."""
    assert (
        _cluster_is_informative(
            mean_per_class={1: 1.0},
            covariance_per_class={0: 0.0, 1: 0.8},
            baseline_correlation=0.0,
            n_classes=2,
        )
        is True
    )


# ---------------------------------------------------------------------------
# Headline end-to-end: a covariance-only noise anchor is derived informative
# ---------------------------------------------------------------------------
def _covariance_only_noise_anchor_config():
    """Build a config whose only cluster is a covariance-only noise anchor.

    The cluster carries a class-varying correlation but no mean channel, so it is
    discriminative through differential co-expression alone.
    """
    return DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=1, class_sep=1.0)],
        n_standalone_noise=1,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                baseline_correlation=0.0,
                covariance_channel=CovarianceChannel(per_class_correlation={0: 0.0, 1: 0.8}),
            ),
        ],
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],
        random_state=42,
    )


def test_covariance_only_noise_anchor_relabeled_informative():
    """The covariance-only noise anchor and its proxies are labeled informative."""
    cfg = _covariance_only_noise_anchor_config()
    _, _, meta = generate_dataset(cfg, return_dataframe=False)
    roles = compute_feature_roles(meta)

    anchor_col = meta.anchor_idx[0]
    proxy_cols = [c for c in meta.corr_cluster_indices[0] if c != anchor_col]

    assert anchor_col in meta.informative_idx
    assert anchor_col in roles.informative_anchor_indices
    assert all(c in roles.informative_proxy_indices for c in proxy_cols)
    assert roles.noise_anchor_indices == []
    assert roles.noise_proxy_indices == []
    # The whole cluster block stays out of the noise index list.
    assert all(c not in meta.noise_idx for c in meta.corr_cluster_indices[0])


def test_covariance_only_noise_anchor_partition_and_consistency():
    """Roles tile every column once and agree with the informative labels."""
    cfg = _covariance_only_noise_anchor_config()
    _, _, meta = generate_dataset(cfg, return_dataframe=False)
    roles = compute_feature_roles(meta)

    role_lists = [
        roles.standalone_informative_indices,
        roles.informative_anchor_indices,
        roles.informative_proxy_indices,
        roles.standalone_noise_indices,
        roles.noise_anchor_indices,
        roles.noise_proxy_indices,
    ]
    union = set()
    total = 0
    for role_list in role_lists:
        union |= set(role_list)
        total += len(role_list)

    assert union == set(range(cfg.n_features))
    # Equal union size and total length implies pairwise disjoint.
    assert total == cfg.n_features

    assert set(meta.informative_idx).isdisjoint(meta.noise_idx)
    for anchor_col in roles.informative_anchor_indices:
        assert anchor_col in meta.informative_idx


# ---------------------------------------------------------------------------
# Step 8.1 -- Standalone-informative group validation
# ---------------------------------------------------------------------------
def _two_classes() -> list[ClassConfig]:
    """Two minimal classes (n_classes - 1 == 1 separation boundary)."""
    return [ClassConfig(n_samples=20), ClassConfig(n_samples=20)]


def test_group_n_features_below_one_raises():
    """A group must declare at least one feature (n_features >= 1)."""
    with pytest.raises(ValidationError):
        StandaloneInformativeGroup(n_features=0, class_sep=1.0)


def test_group_sequence_class_sep_wrong_length_raises():
    """A sequence class_sep must have length n_classes - 1 (here 1)."""
    with pytest.raises(ValidationError, match="class_sep has length"):
        DatasetConfig(
            standalone_informative_groups=[StandaloneInformativeGroup(n_features=2, class_sep=[1.0, 2.0])],
            class_configs=_two_classes(),
            random_state=0,
        )


def test_group_scalar_class_sep_accepted_for_any_n_classes():
    """A scalar class_sep broadcasts and is accepted for 2 and 4 classes alike."""
    for n_classes in (2, 4):
        cfg = DatasetConfig(
            standalone_informative_groups=[StandaloneInformativeGroup(n_features=3, class_sep=1.5)],
            class_configs=[ClassConfig(n_samples=10) for _ in range(n_classes)],
            random_state=0,
        )
        assert cfg.n_classes == n_classes


def test_group_non_finite_class_sep_raises():
    """A non-finite scalar class_sep is rejected at group construction."""
    with pytest.raises(ValidationError, match="finite"):
        StandaloneInformativeGroup(n_features=1, class_sep=float("inf"))


# ---------------------------------------------------------------------------
# Step 8.2 -- Derived counts (no declared budget asserted)
# ---------------------------------------------------------------------------
def test_derived_counts_from_groups_and_clusters():
    """n_standalone_informative is the group-size sum; n_features is their total."""
    cfg = DatasetConfig(
        standalone_informative_groups=[
            StandaloneInformativeGroup(n_features=2, class_sep=2.0),
            StandaloneInformativeGroup(n_features=3, class_sep=0.5),
        ],
        n_standalone_noise=4,
        corr_clusters=[
            CorrClusterConfig(n_cluster_features=3, mean_channel=MeanChannel(per_class_effect={1: 1.0})),
            CorrClusterConfig(n_cluster_features=2, baseline_correlation=0.5),
        ],
        class_configs=_two_classes(),
        random_state=0,
    )
    assert cfg.n_standalone_informative == 2 + 3
    cluster_members = 3 + 2
    assert cfg.n_features == cfg.n_standalone_informative + cfg.n_standalone_noise + cluster_members


# ---------------------------------------------------------------------------
# Step 8.3 -- Gradient across standalone-informative groups
# ---------------------------------------------------------------------------
def test_standalone_group_strength_gradient():
    """Two groups of different class_sep give a per-group constant strength gradient."""
    cfg = DatasetConfig(
        standalone_informative_groups=[
            StandaloneInformativeGroup(n_features=2, class_sep=2.0),
            StandaloneInformativeGroup(n_features=3, class_sep=0.5),
        ],
        class_configs=_two_classes(),
        random_state=0,
    )
    _, _, meta = generate_dataset(cfg, return_dataframe=False)
    strengths = compute_feature_strengths(meta)

    group_high, group_low = meta.standalone_informative_groups
    high_vals = [strengths.mean_strength[c] for c in group_high.column_indices]
    low_vals = [strengths.mean_strength[c] for c in group_low.column_indices]

    # Constant within each group's column range.
    assert all(abs(v - high_vals[0]) <= _TOL for v in high_vals)
    assert all(abs(v - low_vals[0]) <= _TOL for v in low_vals)
    # Distinct across ranges, ordered consistently with declaration order
    # (the first, larger-class_sep group carries the stronger signal).
    assert high_vals[0] > low_vals[0] + _TOL


# ---------------------------------------------------------------------------
# Step 8.4 -- Cluster mean strength: anchor range and proxy propagation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("structure", ["equicorrelated", "toeplitz"])
def test_cluster_mean_strength_anchor_and_proxy_propagation(structure):
    """Anchor strength is the mean-channel range; proxies match _proxy_mean_offset."""
    effect = 2.0
    baseline = 0.5
    cfg = DatasetConfig(
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                correlation_structure=structure,
                baseline_correlation=baseline,
                anchor_index=0,
                mean_channel=MeanChannel(per_class_effect={1: effect}),
            ),
        ],
        class_configs=_two_classes(),
        random_state=0,
    )
    _, _, meta = generate_dataset(cfg, return_dataframe=False)
    strengths = compute_feature_strengths(meta)

    anchor_col = meta.anchor_idx[0]
    # Anchor range over classes: {0: 0.0, 1: effect} -> effect - 0.0.
    assert abs(strengths.mean_strength[anchor_col] - effect) <= _TOL

    for proxy_col in meta.corr_cluster_indices[0]:
        if proxy_col == anchor_col:
            continue
        distance = abs(proxy_col - anchor_col)
        # No covariance channel: effective per-class correlation is the baseline.
        expected = _proxy_mean_offset(
            anchor_per_class_offset=effect,
            distance=distance,
            correlation_structure=structure,
            effective_per_class_correlation=baseline,
            proxy_attenuation=1.0,
        )
        assert abs(strengths.mean_strength[proxy_col] - expected) <= _TOL


def test_cluster_mean_strength_per_class_attenuation():
    """With a per-class covariance channel, each class propagates at its own rho."""
    effect = 1.5
    per_class_rho = {0: 0.2, 1: 0.6}
    attenuation = 0.8
    cfg = DatasetConfig(
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=2,
                correlation_structure="equicorrelated",
                baseline_correlation=0.3,
                anchor_index=0,
                proxy_attenuation=attenuation,
                mean_channel=MeanChannel(per_class_effect={1: effect}),
                covariance_channel=CovarianceChannel(per_class_correlation=per_class_rho),
            ),
        ],
        class_configs=_two_classes(),
        random_state=0,
    )
    _, _, meta = generate_dataset(cfg, return_dataframe=False)
    strengths = compute_feature_strengths(meta)

    anchor_col = meta.anchor_idx[0]
    (proxy_col,) = [c for c in meta.corr_cluster_indices[0] if c != anchor_col]
    distance = abs(proxy_col - anchor_col)

    # Per-class propagated offsets resolve each class at its own effective rho.
    offset_class0 = _proxy_mean_offset(0.0, distance, "equicorrelated", per_class_rho[0], attenuation)
    offset_class1 = _proxy_mean_offset(effect, distance, "equicorrelated", per_class_rho[1], attenuation)
    expected_proxy = max(offset_class0, offset_class1) - min(offset_class0, offset_class1)
    assert abs(strengths.mean_strength[proxy_col] - expected_proxy) <= _TOL


# ---------------------------------------------------------------------------
# Step 8.5 -- Cluster covariance strength
# ---------------------------------------------------------------------------
def test_cluster_covariance_strength_range_and_uniform_within_cluster():
    """covariance_strength is the per-class correlation range, shared cluster-wide."""
    per_class_rho = {0: 0.1, 1: 0.7}
    cfg = DatasetConfig(
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                baseline_correlation=0.4,
                covariance_channel=CovarianceChannel(per_class_correlation=per_class_rho),
            ),
        ],
        class_configs=_two_classes(),
        random_state=0,
    )
    _, _, meta = generate_dataset(cfg, return_dataframe=False)
    strengths = compute_feature_strengths(meta)

    expected_range = max(per_class_rho.values()) - min(per_class_rho.values())
    cluster_cols = meta.corr_cluster_indices[0]
    for col in cluster_cols:
        assert abs(strengths.covariance_strength[col] - expected_range) <= _TOL
    # Identical for anchor and every proxy.
    values = {strengths.covariance_strength[c] for c in cluster_cols}
    assert len(values) == 1


def test_cluster_covariance_strength_uses_baseline_for_absent_class():
    """An absent class falls back to baseline_correlation when forming the range."""
    baseline = 0.5
    cfg = DatasetConfig(
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=2,
                baseline_correlation=baseline,
                # Class 1 absent -> resolves to baseline; class 0 set to 0.0.
                covariance_channel=CovarianceChannel(per_class_correlation={0: 0.0}),
            ),
        ],
        class_configs=_two_classes(),
        random_state=0,
    )
    _, _, meta = generate_dataset(cfg, return_dataframe=False)
    strengths = compute_feature_strengths(meta)
    for col in meta.corr_cluster_indices[0]:
        assert abs(strengths.covariance_strength[col] - baseline) <= _TOL


# ---------------------------------------------------------------------------
# Step 8.6 -- Consistency between strengths, channels, and roles
# ---------------------------------------------------------------------------
def _mixed_channels_config() -> DatasetConfig:
    """One mean-only, one covariance-only, and one mixed cluster, plus noise."""
    return DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=2, class_sep=1.0)],
        n_standalone_noise=2,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=2,
                # Nonzero baseline so the proxy inherits the anchor's mean signal
                # (a zero correlation would zero out the propagated proxy offset).
                baseline_correlation=0.5,
                mean_channel=MeanChannel(per_class_effect={1: 1.0}),
            ),
            CorrClusterConfig(
                n_cluster_features=2,
                baseline_correlation=0.0,
                covariance_channel=CovarianceChannel(per_class_correlation={0: 0.0, 1: 0.8}),
            ),
            CorrClusterConfig(
                n_cluster_features=2,
                baseline_correlation=0.3,
                mean_channel=MeanChannel(per_class_effect={1: 1.2}),
                covariance_channel=CovarianceChannel(per_class_correlation={0: 0.2, 1: 0.6}),
            ),
            # A genuine noise cluster: constant correlation, no mean channel.
            CorrClusterConfig(n_cluster_features=2, baseline_correlation=0.5),
        ],
        class_configs=_two_classes(),
        random_state=0,
    )


def test_signal_channels_match_informative_roles():
    """Columns with non-empty channels equal the informative columns from roles."""
    cfg = _mixed_channels_config()
    _, _, meta = generate_dataset(cfg, return_dataframe=False)
    strengths = compute_feature_strengths(meta)
    roles = compute_feature_roles(meta)

    columns_with_signal = {i for i, ch in enumerate(strengths.signal_channels) if ch}
    informative_from_roles = set(
        roles.standalone_informative_indices + roles.informative_anchor_indices + roles.informative_proxy_indices
    )
    assert columns_with_signal == informative_from_roles


def test_channel_sets_per_cluster_kind():
    """mean-only -> {mean}; covariance-only -> {covariance}; mixed -> both."""
    cfg = _mixed_channels_config()
    _, _, meta = generate_dataset(cfg, return_dataframe=False)
    strengths = compute_feature_strengths(meta)

    mean_only_cols = meta.corr_cluster_indices[0]
    cov_only_cols = meta.corr_cluster_indices[1]
    mixed_cols = meta.corr_cluster_indices[2]

    for col in mean_only_cols:
        assert set(strengths.signal_channels[col]) == {"mean"}
    for col in cov_only_cols:
        assert set(strengths.signal_channels[col]) == {"covariance"}
    for col in mixed_cols:
        assert set(strengths.signal_channels[col]) == {"covariance", "mean"}


def test_member_strength_positive_iff_informative_role():
    """Each cluster member has positive strength iff it sits in an informative role."""
    cfg = _mixed_channels_config()
    _, _, meta = generate_dataset(cfg, return_dataframe=False)
    strengths = compute_feature_strengths(meta)
    roles = compute_feature_roles(meta)

    informative_members = set(roles.informative_anchor_indices + roles.informative_proxy_indices)
    noise_members = set(roles.noise_anchor_indices + roles.noise_proxy_indices)

    for col in informative_members | noise_members:
        has_signal = strengths.mean_strength[col] > 0.0 or strengths.covariance_strength[col] > 0.0
        assert has_signal == (col in informative_members)


# ---------------------------------------------------------------------------
# Step 8.7 -- Noise features carry no strength
# ---------------------------------------------------------------------------
def test_noise_features_have_zero_strength_and_empty_channels():
    """Standalone-noise, noise-anchor, and noise-proxy columns are all signal-free."""
    cfg = _mixed_channels_config()
    _, _, meta = generate_dataset(cfg, return_dataframe=False)
    strengths = compute_feature_strengths(meta)
    roles = compute_feature_roles(meta)

    noise_cols = roles.standalone_noise_indices + roles.noise_anchor_indices + roles.noise_proxy_indices
    # The config includes all three noise kinds.
    assert roles.standalone_noise_indices
    assert roles.noise_anchor_indices
    assert roles.noise_proxy_indices

    for col in noise_cols:
        assert strengths.mean_strength[col] == 0.0
        assert strengths.covariance_strength[col] == 0.0
        assert strengths.signal_channels[col] == ()


# ---------------------------------------------------------------------------
# Step 8.8 -- Full per-column coverage
# ---------------------------------------------------------------------------
def test_strengths_cover_every_column_exactly_once():
    """All three sequences have length n_features and index every column once."""
    cfg = _mixed_channels_config()
    _, _, meta = generate_dataset(cfg, return_dataframe=False)
    strengths = compute_feature_strengths(meta)

    n_features = cfg.n_features
    assert len(strengths.mean_strength) == n_features
    assert len(strengths.covariance_strength) == n_features
    assert len(strengths.signal_channels) == n_features

    # Positional sequences: a value per column 0..n_features-1, exactly once.
    assert n_features == len(meta.feature_names)
    role_cols = set(range(n_features))
    assert role_cols == set(range(len(strengths.mean_strength)))


# ---------------------------------------------------------------------------
# Per-column relevance: role assignment, counts, and strengths are one predicate
# ---------------------------------------------------------------------------
def test_informative_role_iff_nonempty_signal_channels():
    """A column is in an informative role iff its signal_channels are non-empty.

    Reproduces the divergence for a mean-only cluster at the default
    baseline_correlation of 0.0: the proxies inherit no propagated signal, yet
    cluster-level role assignment places them in informative_proxy_indices.
    """
    cfg = DatasetConfig(
        standalone_informative_groups=[],
        n_standalone_noise=0,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                baseline_correlation=0.0,  # the field default
                mean_channel=MeanChannel(per_class_effect={1: 1.0}),
            ),
        ],
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],
        random_state=0,
    )

    _, _, meta = generate_dataset(cfg, return_dataframe=False)
    roles = compute_feature_roles(meta)
    strengths = compute_feature_strengths(meta)

    informative_role_columns = set(
        roles.standalone_informative_indices
        + roles.informative_anchor_indices
        + roles.informative_proxy_indices
    )
    for column_index in range(cfg.n_features):
        in_informative_role = column_index in informative_role_columns
        has_signal = len(strengths.signal_channels[column_index]) > 0
        assert in_informative_role == has_signal, (
            f"column {column_index}: informative_role={in_informative_role}, "
            f"signal_channels={strengths.signal_channels[column_index]}"
        )


def test_mean_only_zero_correlation_proxies_are_noise():
    """Mean-only, zero within-cluster correlation: only the anchor carries signal."""
    cfg = DatasetConfig(
        standalone_informative_groups=[],
        n_standalone_noise=0,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                baseline_correlation=0.0,
                mean_channel=MeanChannel(per_class_effect={1: 1.0}),
            ),
        ],
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],
        random_state=0,
    )
    _, _, meta = generate_dataset(cfg, return_dataframe=False)
    roles = compute_feature_roles(meta)

    anchor_col = meta.anchor_idx[0]
    proxy_cols = [c for c in meta.corr_cluster_indices[0] if c != anchor_col]

    assert roles.informative_anchor_indices == [anchor_col]
    assert sorted(roles.noise_proxy_indices) == sorted(proxy_cols)
    assert roles.informative_proxy_indices == []
    assert roles.noise_anchor_indices == []
    assert cfg.n_informative == 1
    assert cfg.n_noise == cfg.n_features - 1
    assert anchor_col in meta.informative_idx
    assert all(c in meta.noise_idx for c in proxy_cols)


def test_mean_cluster_with_correlation_keeps_all_members_informative():
    """Regression guard: nonzero correlation -> no proxy demotion."""
    cfg = DatasetConfig(
        standalone_informative_groups=[],
        n_standalone_noise=0,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                baseline_correlation=0.6,
                mean_channel=MeanChannel(per_class_effect={1: 1.0}),
            ),
        ],
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],
        random_state=0,
    )
    _, _, meta = generate_dataset(cfg, return_dataframe=False)
    roles = compute_feature_roles(meta)
    assert roles.noise_proxy_indices == []
    assert roles.noise_anchor_indices == []
    assert cfg.n_informative == 3


def test_covariance_only_cluster_keeps_all_members_informative():
    """Regression guard: the second-moment signal is shared cluster-wide."""
    cfg = DatasetConfig(
        standalone_informative_groups=[],
        n_standalone_noise=0,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                baseline_correlation=0.0,
                covariance_channel=CovarianceChannel(per_class_correlation={0: 0.0, 1: 0.8}),
            ),
        ],
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],
        random_state=0,
    )
    _, _, meta = generate_dataset(cfg, return_dataframe=False)
    roles = compute_feature_roles(meta)
    assert roles.noise_proxy_indices == []
    assert roles.noise_anchor_indices == []
    assert cfg.n_informative == 3


# ---------------------------------------------------------------------------
# Parametrized invariant sweep: (informative role) == (non-empty channels)
# ---------------------------------------------------------------------------
def _invariant_sweep_clusters() -> dict[str, CorrClusterConfig]:
    """Representative single-cluster configs covering the relevant channel mixes."""
    return {
        "mean_only_rho0": CorrClusterConfig(
            n_cluster_features=3,
            baseline_correlation=0.0,
            mean_channel=MeanChannel(per_class_effect={1: 1.0}),
        ),
        "mean_only_rho06": CorrClusterConfig(
            n_cluster_features=3,
            baseline_correlation=0.6,
            mean_channel=MeanChannel(per_class_effect={1: 1.0}),
        ),
        "covariance_only": CorrClusterConfig(
            n_cluster_features=3,
            baseline_correlation=0.0,
            covariance_channel=CovarianceChannel(per_class_correlation={0: 0.0, 1: 0.8}),
        ),
        "mean_covariance_mixed": CorrClusterConfig(
            n_cluster_features=3,
            baseline_correlation=0.3,
            mean_channel=MeanChannel(per_class_effect={1: 1.2}),
            covariance_channel=CovarianceChannel(per_class_correlation={0: 0.2, 1: 0.6}),
        ),
        "pure_noise": CorrClusterConfig(
            n_cluster_features=3,
            baseline_correlation=0.5,
        ),
        "toeplitz_mean": CorrClusterConfig(
            n_cluster_features=4,
            correlation_structure="toeplitz",
            baseline_correlation=0.5,
            mean_channel=MeanChannel(per_class_effect={1: 1.0}),
        ),
    }


@pytest.mark.parametrize("cluster_key", list(_invariant_sweep_clusters().keys()))
def test_informative_role_iff_nonempty_signal_channels_sweep(cluster_key):
    """(column in an informative role) == (signal_channels non-empty), per cluster config."""
    cluster = _invariant_sweep_clusters()[cluster_key]
    cfg = DatasetConfig(
        standalone_informative_groups=[],
        n_standalone_noise=0,
        corr_clusters=[cluster],
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],
        random_state=0,
    )

    _, _, meta = generate_dataset(cfg, return_dataframe=False)
    roles = compute_feature_roles(meta)
    strengths = compute_feature_strengths(meta)

    informative_role_columns = set(
        roles.standalone_informative_indices
        + roles.informative_anchor_indices
        + roles.informative_proxy_indices
    )
    for column_index in range(cfg.n_features):
        in_informative_role = column_index in informative_role_columns
        has_signal = len(strengths.signal_channels[column_index]) > 0
        assert in_informative_role == has_signal, (
            f"column {column_index}: informative_role={in_informative_role}, "
            f"signal_channels={strengths.signal_channels[column_index]}"
        )
