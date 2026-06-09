# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Extended tests for configuration validation."""

import pytest

from biomedical_data_generator import (
    ClassConfig,
    CorrClusterConfig,
    CovarianceChannel,
    DatasetConfig,
    MeanChannel,
    StandaloneInformativeGroup,
)
from biomedical_data_generator.config import (
    BatchEffectsConfig,
    validate_distribution_params,
)


# ============================================================================
# Tests for validate_distribution_params
# ============================================================================
def test_validate_distribution_params_empty():
    """Test validation with empty params dict."""
    result = validate_distribution_params({}, "normal")
    assert result == {}


def test_validate_distribution_params_normal_valid():
    """Test validation for normal distribution."""
    params = {"loc": 0.0, "scale": 1.0}
    result = validate_distribution_params(params, "normal")
    assert result == params


def test_validate_distribution_params_normal_invalid_key():
    """Test that invalid parameter keys raise ValueError."""
    params = {"loc": 0.0, "invalid_param": 1.0}

    with pytest.raises(ValueError, match="Invalid parameters"):
        validate_distribution_params(params, "normal")


def test_validate_distribution_params_uniform_missing_required():
    """Test that missing required params raise ValueError."""
    params = {"low": 0.0}  # Missing 'high'

    with pytest.raises(ValueError, match="Missing required parameters"):
        validate_distribution_params(params, "uniform")


def test_validate_distribution_params_uniform_invalid_range():
    """Test that high <= low raises ValueError."""
    params = {"low": 5.0, "high": 3.0}

    with pytest.raises(ValueError, match="'high' .* must be > 'low'"):
        validate_distribution_params(params, "uniform")


def test_validate_distribution_params_uniform_equal_values():
    """Test that equal low and high raises ValueError."""
    params = {"low": 5.0, "high": 5.0}

    with pytest.raises(ValueError, match="'high' .* must be > 'low'"):
        validate_distribution_params(params, "uniform")


def test_validate_distribution_params_uniform_non_numeric():
    """Test that non-numeric values raise ValueError."""
    params = {"low": "not_a_number", "high": 10.0}

    with pytest.raises(ValueError, match="must be numeric"):
        validate_distribution_params(params, "uniform")


def test_validate_distribution_params_scale_negative():
    """Test that negative scale raises ValueError."""
    params = {"loc": 0.0, "scale": -1.0}

    with pytest.raises(ValueError, match="'scale' must be > 0"):
        validate_distribution_params(params, "normal")


def test_validate_distribution_params_scale_zero():
    """Test that zero scale raises ValueError."""
    params = {"scale": 0.0}

    with pytest.raises(ValueError, match="'scale' must be > 0"):
        validate_distribution_params(params, "exponential")


def test_validate_distribution_params_scale_non_numeric():
    """Test that non-numeric scale raises ValueError."""
    params = {"scale": "not_a_number"}

    with pytest.raises(ValueError, match="'scale' must be numeric"):
        validate_distribution_params(params, "normal")


def test_validate_distribution_params_sigma_negative():
    """Test that negative sigma raises ValueError for lognormal."""
    params = {"sigma": -0.5}

    with pytest.raises(ValueError, match="'sigma' must be > 0"):
        validate_distribution_params(params, "lognormal")


def test_validate_distribution_params_sigma_non_numeric():
    """Test that non-numeric sigma raises ValueError."""
    params = {"sigma": "not_a_number"}

    with pytest.raises(ValueError, match="'sigma' must be numeric"):
        validate_distribution_params(params, "lognormal")


def test_validate_distribution_params_loc_non_numeric():
    """Test that non-numeric loc raises ValueError."""
    params = {"loc": "not_a_number", "scale": 1.0}

    with pytest.raises(ValueError, match="'loc' must be numeric"):
        validate_distribution_params(params, "normal")


def test_validate_distribution_params_mean_non_numeric():
    """Test that non-numeric mean raises ValueError."""
    params = {"mean": "not_a_number"}

    with pytest.raises(ValueError, match="'mean' must be numeric"):
        validate_distribution_params(params, "lognormal")


def test_validate_distribution_params_unsupported_distribution():
    """Test that unsupported distribution returns params unchanged."""
    params = {"some_param": 1.0}
    result = validate_distribution_params(params, "unsupported_dist")
    assert result == params


# ============================================================================
# Tests for ClassConfig
# ===========================================================================
def test_class_config_default_values():
    """Test ClassConfig with default values."""
    cfg = ClassConfig()

    assert cfg.n_samples == 30
    assert cfg.class_distribution == "normal"
    assert cfg.class_distribution_params == {"loc": 0, "scale": 1}
    assert cfg.label is None


def test_class_config_custom_values():
    """Test ClassConfig with custom values."""
    cfg = ClassConfig(
        n_samples=100,
        class_distribution="uniform",
        class_distribution_params={"low": 0.0, "high": 1.0},
        label="healthy",
    )

    assert cfg.n_samples == 100
    assert cfg.class_distribution == "uniform"
    assert cfg.label == "healthy"


def test_class_config_invalid_n_samples():
    """Test that n_samples < 1 raises error."""
    with pytest.raises(ValueError):
        ClassConfig(n_samples=0)


def test_class_config_str_representation():
    """Test __str__ method of ClassConfig."""
    cfg = ClassConfig(n_samples=50, label="test")
    s = str(cfg)

    assert "n=50" in s
    assert "label='test'" in s


def test_class_config_str_with_non_default_distribution():
    """Test __str__ includes distribution when non-default."""
    cfg = ClassConfig(n_samples=50, class_distribution="lognormal")
    s = str(cfg)

    assert "dist=lognormal" in s


# ============================================================================
# Tests for CorrClusterConfig
# ============================================================================
def test_corr_cluster_config_defaults():
    """Test CorrClusterConfig with default values."""
    cfg = CorrClusterConfig(n_cluster_features=5)

    assert cfg.n_cluster_features == 5
    assert cfg.baseline_correlation == 0.0
    assert cfg.correlation_structure == "equicorrelated"
    assert cfg.anchor_index == 0
    assert cfg.proxy_attenuation == 1.0
    assert cfg.mean_channel is None
    assert cfg.covariance_channel is None


def test_corr_cluster_config_single_feature_rejected():
    """A single-feature cluster is rejected: correlation is undefined for p=1."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="greater than or equal to 2"):
        CorrClusterConfig(n_cluster_features=1)


def test_corr_cluster_config_covariance_channel_class_specific():
    """A covariance channel resolves a per-class within-cluster correlation."""
    cfg = CorrClusterConfig(
        n_cluster_features=5,
        baseline_correlation=0.1,
        covariance_channel=CovarianceChannel(per_class_correlation={0: 0.8, 1: 0.6}),
    )

    assert cfg.effective_correlation_for_class(0) == 0.8
    assert cfg.effective_correlation_for_class(1) == 0.6
    # Unspecified class falls back to baseline_correlation.
    assert cfg.effective_correlation_for_class(2) == 0.1


def test_corr_cluster_config_covariance_channel_out_of_range():
    """A covariance channel correlation outside the valid range raises an error."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="covariance_channel"):
        CorrClusterConfig(
            n_cluster_features=5,
            covariance_channel=CovarianceChannel(per_class_correlation={0: 0.0, 1: 1.4}),
        )


def test_corr_cluster_config_baseline_correlation_out_of_range():
    """Test that baseline_correlation outside [-1, 1] raises error."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="baseline_correlation"):
        CorrClusterConfig(
            n_cluster_features=5,
            baseline_correlation=1.5,
        )


def test_corr_cluster_config_correlation_equicorrelated_boundary():
    """Test boundary check for equicorrelated structure."""
    from pydantic import ValidationError

    # For equicorrelated, correlation must be > -1/(n-1).
    # For n=5, lower bound is -1/4 = -0.25, so -0.3 is too negative.
    with pytest.raises(ValidationError, match="baseline_correlation"):
        CorrClusterConfig(
            n_cluster_features=5,
            baseline_correlation=-0.3,  # Too negative
            correlation_structure="equicorrelated",
        )


def test_corr_cluster_config_toeplitz_structure():
    """Test toeplitz structure validation."""
    cfg = CorrClusterConfig(
        n_cluster_features=5,
        baseline_correlation=0.7,
        correlation_structure="toeplitz",
    )

    assert cfg.correlation_structure == "toeplitz"


def test_corr_cluster_config_toeplitz_correlation_boundary():
    """Test that correlation at boundary for toeplitz raises error."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="baseline_correlation"):
        CorrClusterConfig(
            n_cluster_features=5,
            baseline_correlation=1.0,
            correlation_structure="toeplitz",
        )


def test_corr_cluster_config_mean_channel_resolution():
    """A mean channel resolves a per-class anchor mean shift."""
    cfg = CorrClusterConfig(
        n_cluster_features=5,
        baseline_correlation=0.7,
        mean_channel=MeanChannel(per_class_effect={1: 1.5}),
    )

    assert cfg.mean_effect_for_class(1) == 1.5
    # Absent classes get the 0.0 baseline shift.
    assert cfg.mean_effect_for_class(0) == 0.0


def test_corr_cluster_config_no_mean_channel_resolves_to_zero():
    """Without a mean channel, every class resolves to a 0.0 mean shift."""
    cfg = CorrClusterConfig(
        n_cluster_features=3,
        baseline_correlation=0.8,
    )
    assert cfg.mean_effect_for_class(0) == 0.0
    assert cfg.mean_effect_for_class(1) == 0.0


def test_corr_cluster_config_proxy_attenuation_custom():
    """proxy_attenuation is a neutral multiplier with a custom value accepted."""
    cfg = CorrClusterConfig(
        n_cluster_features=4,
        baseline_correlation=0.5,
        proxy_attenuation=0.25,
    )
    assert cfg.proxy_attenuation == 0.25


def test_corr_cluster_config_anchor_index_out_of_range():
    """Test that an anchor_index outside the block raises error."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="anchor_index"):
        CorrClusterConfig(
            n_cluster_features=5,
            anchor_index=5,  # valid indices are 0..4
        )


# ============================================================================
# Tests for BatchEffectsConfig
# ============================================================================
def test_batch_effects_config_defaults():
    """Test BatchEffectsConfig default values."""
    cfg = BatchEffectsConfig()

    assert cfg.n_batches == 0  # Default is 0, not 3
    assert cfg.effect_strength == 0.5  # Default is 0.5, not 1.0
    assert cfg.effect_type == "additive"
    assert cfg.confounding_with_class == 0.0


def test_batch_effects_config_custom_values():
    """Test BatchEffectsConfig with custom values."""
    cfg = BatchEffectsConfig(
        n_batches=5,
        effect_strength=2.0,
        effect_type="multiplicative",
        confounding_with_class=0.7,
    )

    assert cfg.n_batches == 5
    assert cfg.effect_strength == 2.0
    assert cfg.effect_type == "multiplicative"
    assert cfg.confounding_with_class == 0.7


def test_batch_effects_config_proportions():
    """Test BatchEffectsConfig with custom proportions."""
    cfg = BatchEffectsConfig(
        n_batches=3,
        proportions=(0.5, 0.3, 0.2),
    )

    # Proportions might be stored as list, not tuple
    assert list(cfg.proportions) == [0.5, 0.3, 0.2]


# ============================================================================
# Tests for DatasetConfig integration
# ============================================================================


def test_dataset_config_with_batch_effects():
    """Test DatasetConfig with batch effects."""
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=5, class_sep=1.0)],
        n_standalone_noise=3,
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],
        batch_effects=BatchEffectsConfig(n_batches=2),
    )

    assert cfg.batch_effects is not None
    assert cfg.batch_effects.n_batches == 2


def test_dataset_config_breakdown():
    """Test breakdown() method returns correct structure."""
    cfg = DatasetConfig(
        standalone_informative_groups=[StandaloneInformativeGroup(n_features=5, class_sep=1.0)],
        n_standalone_noise=3,
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],
    )

    breakdown = cfg.breakdown()
    assert "n_features" in breakdown
    assert "n_informative" in breakdown
    assert "n_noise" in breakdown
    # No clusters here, so derived counts equal the standalone counts.
    assert breakdown["n_informative"] == 5
    assert breakdown["n_noise"] == 3
    assert breakdown["n_features"] == 8


def test_dataset_config_extra_fields_forbidden():
    """Test that extra fields are rejected."""
    with pytest.raises(ValueError):
        DatasetConfig(
            standalone_informative_groups=[StandaloneInformativeGroup(n_features=5, class_sep=1.0)],
            n_standalone_noise=3,
            class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],
            unknown_field="value",
        )
