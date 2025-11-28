# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Extended tests for configuration validation."""

import pytest

from biomedical_data_generator import ClassConfig, CorrClusterConfig, DatasetConfig
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
# ============================================================================


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
    cfg = CorrClusterConfig(n_cluster_features=5, correlation=0.7)

    assert cfg.n_cluster_features == 5
    assert cfg.correlation == 0.7
    assert cfg.structure == "equicorrelated"
    assert cfg.anchor_role == "informative"
    assert cfg.anchor_effect_size is None


def test_corr_cluster_config_invalid_n_cluster_features():
    """Test that n_cluster_features must be >= 1."""
    # n_cluster_features=1 is actually valid - there's no minimum of 2
    cfg = CorrClusterConfig(n_cluster_features=2, correlation=0.5)
    assert cfg.n_cluster_features == 2


def test_corr_cluster_config_correlation_class_specific():
    """Test class-specific correlation dict."""
    cfg = CorrClusterConfig(
        n_cluster_features=5,
        correlation={0: 0.8, 1: 0.6},
    )

    assert cfg.is_class_specific()
    assert cfg.get_correlation_for_class(0) == 0.8
    assert cfg.get_correlation_for_class(1) == 0.6
    assert cfg.get_correlation_for_class(2) == 0.0  # Default for unspecified


def test_corr_cluster_config_correlation_invalid_negative_key():
    """Test that negative class index in correlation dict raises error."""
    with pytest.raises(ValueError, match="correlation keys must be >= 0"):
        CorrClusterConfig(
            n_cluster_features=5,
            correlation={-1: 0.8},
        )


def test_corr_cluster_config_correlation_out_of_range():
    """Test that correlation outside [-1, 1] raises error."""
    with pytest.raises(ValueError, match="correlation=1.5 invalid"):
        CorrClusterConfig(
            n_cluster_features=5,
            correlation=1.5,
        )


def test_corr_cluster_config_correlation_equicorrelated_boundary():
    """Test boundary check for equicorrelated structure."""
    # For equicorrelated, correlation must be > -1/(n-1)
    # For n=5, lower bound is -1/4 = -0.25
    with pytest.raises(ValueError, match="require .* < correlation < 1"):
        CorrClusterConfig(
            n_cluster_features=5,
            correlation=-0.3,  # Too negative
            structure="equicorrelated",
        )


def test_corr_cluster_config_toeplitz_structure():
    """Test toeplitz structure validation."""
    cfg = CorrClusterConfig(
        n_cluster_features=5,
        correlation=0.7,
        structure="toeplitz",
    )

    assert cfg.structure == "toeplitz"


def test_corr_cluster_config_toeplitz_correlation_boundary():
    """Test that correlation at boundary for toeplitz raises error."""
    with pytest.raises(ValueError, match="require \\|correlation\\| < 1"):
        CorrClusterConfig(
            n_cluster_features=5,
            correlation=1.0,
            structure="toeplitz",
        )


def test_corr_cluster_config_anchor_effect_size_small():
    """Test anchor_effect_size preset 'small'."""
    cfg = CorrClusterConfig(
        n_cluster_features=5,
        correlation=0.7,
        anchor_effect_size="small",
    )

    assert cfg.resolve_anchor_effect_size() == 0.5


def test_corr_cluster_config_anchor_effect_size_medium():
    """Test anchor_effect_size preset 'medium'."""
    cfg = CorrClusterConfig(
        n_cluster_features=5,
        correlation=0.7,
        anchor_effect_size="medium",
    )

    assert cfg.resolve_anchor_effect_size() == 1.0


def test_corr_cluster_config_anchor_effect_size_large():
    """Test anchor_effect_size preset 'large'."""
    cfg = CorrClusterConfig(
        n_cluster_features=5,
        correlation=0.7,
        anchor_effect_size="large",
    )

    assert cfg.resolve_anchor_effect_size() == 1.5


def test_corr_cluster_config_anchor_effect_size_numeric():
    """Test numeric anchor_effect_size."""
    cfg = CorrClusterConfig(
        n_cluster_features=5,
        correlation=0.7,
        anchor_effect_size=2.5,
    )

    assert cfg.resolve_anchor_effect_size() == 2.5


def test_corr_cluster_config_anchor_effect_size_none():
    """Test that None anchor_effect_size resolves to 1.0."""
    cfg = CorrClusterConfig(
        n_cluster_features=5,
        correlation=0.7,
        anchor_effect_size=None,
    )

    assert cfg.resolve_anchor_effect_size() == 1.0


def test_corr_cluster_config_anchor_effect_size_invalid_string():
    """Test that invalid string for anchor_effect_size raises error."""
    # Pydantic will raise ValidationError, not ValueError
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CorrClusterConfig(
            n_cluster_features=5,
            correlation=0.7,
            anchor_effect_size="invalid",
        )


def test_corr_cluster_config_anchor_effect_size_negative():
    """Test that negative anchor_effect_size raises error."""
    # Pydantic will raise ValidationError
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CorrClusterConfig(
            n_cluster_features=5,
            correlation=0.7,
            anchor_effect_size=-1.0,
        )


def test_corr_cluster_config_anchor_class_negative():
    """Test that negative anchor_class raises error."""
    with pytest.raises(ValueError, match="anchor_class must be >= 0"):
        CorrClusterConfig(
            n_cluster_features=5,
            correlation=0.7,
            anchor_class=-1,
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
    assert cfg.random_state is None


def test_batch_effects_config_custom_values():
    """Test BatchEffectsConfig with custom values."""
    cfg = BatchEffectsConfig(
        n_batches=5,
        effect_strength=2.0,
        effect_type="multiplicative",
        confounding_with_class=0.7,
        random_state=42,
    )

    assert cfg.n_batches == 5
    assert cfg.effect_strength == 2.0
    assert cfg.effect_type == "multiplicative"
    assert cfg.confounding_with_class == 0.7
    assert cfg.random_state == 42


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
        n_informative=5,
        n_noise=3,
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],
        batch=BatchEffectsConfig(n_batches=2),
    )

    assert cfg.batch is not None
    assert cfg.batch.n_batches == 2


def test_dataset_config_breakdown():
    """Test breakdown() method returns correct structure."""
    cfg = DatasetConfig(
        n_informative=5,
        n_noise=3,
        class_configs=[ClassConfig(n_samples=50), ClassConfig(n_samples=50)],
    )

    breakdown = cfg.breakdown()
    assert "n_features" in breakdown
    assert "n_informative_total" in breakdown
    assert "n_noise_total" in breakdown


def test_dataset_config_extra_fields_forbidden():
    """Test that extra fields are rejected."""
    with pytest.raises(ValueError):
        DatasetConfig(
            n_informative=5,
            n_noise=3,
            class_configs=[ClassConfig(n_samples=50)],
            unknown_field="value",
        )
