# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Configuration models for the dataset generator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .meta import _cluster_column_carries_signal, _cluster_is_informative

DistributionType = Literal[
    "normal",
    "lognormal",
    "exp_normal",  # np.exp(rng.normal()) - direct control over underlying parameters for lognormal distribution
    "uniform",
    "exponential",
    "laplace",
]


# Single source of truth tying each distribution to its allowed parameter
# names and the canonical defaults used when no parameters are provided.
# Defaults mirror the NumPy Generator API (e.g. uniform -> [0, 1)).
_DISTRIBUTION_SCHEMA: dict[str, dict[str, Any]] = {
    "normal": {"required": set(), "optional": {"loc", "scale"}, "defaults": {"loc": 0.0, "scale": 1.0}},
    "lognormal": {"required": set(), "optional": {"mean", "sigma"}, "defaults": {"mean": 0.0, "sigma": 1.0}},
    "exp_normal": {"required": set(), "optional": {"loc", "scale"}, "defaults": {"loc": 0.0, "scale": 1.0}},
    "uniform": {"required": {"low", "high"}, "optional": set(), "defaults": {"low": 0.0, "high": 1.0}},
    "exponential": {"required": set(), "optional": {"scale"}, "defaults": {"scale": 1.0}},
    "laplace": {"required": set(), "optional": {"loc", "scale"}, "defaults": {"loc": 0.0, "scale": 1.0}},
}


def default_distribution_params(distribution: str) -> dict[str, float]:
    """Return the canonical default parameters for a distribution.

    Defaults mirror the NumPy Generator API so that selecting a distribution
    without explicit parameters yields the library's standard behavior (e.g.
    ``uniform`` over ``[0, 1)``, ``normal`` with ``loc=0, scale=1``).

    Args:
        distribution: Distribution type (e.g., "normal", "uniform").

    Returns:
        A fresh dict with the default parameters for the distribution. Returns
        an empty dict for unknown distributions (no parameters assumed).
    """
    schema = _DISTRIBUTION_SCHEMA.get(distribution)
    if schema is None:
        return {}
    # Return a copy so callers cannot mutate the shared schema template.
    return dict(schema["defaults"])


def validate_distribution_params(
    params: dict[str, float],
    distribution: str,
) -> dict[str, float]:
    """Shared validator for distribution parameters.

    Args:
        params: Parameter dict to validate.
        distribution: Distribution type (e.g., "normal", "uniform").

    Returns:
        Validated parameter dict.

    Raises:
        ValueError: If parameters are invalid for the given distribution.
    """
    if not params:
        return params

    schema = _DISTRIBUTION_SCHEMA.get(distribution)
    if not schema:
        return params

    allowed = schema["required"] | schema["optional"]
    provided = set(params.keys())
    invalid = provided - allowed
    if invalid:
        raise ValueError(f"Invalid parameters {invalid} for '{distribution}'. Allowed: {allowed}")

    missing = schema["required"] - provided
    if missing:
        raise ValueError(f"Missing required parameters {missing} for '{distribution}'")

    # Distribution-specific checks
    if distribution == "uniform":
        try:
            low = float(params["low"])
            high = float(params["high"])
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"uniform parameters must be numeric, got low={params['low']}, high={params['high']}"
            ) from e

        if not (low < high):
            raise ValueError(f"uniform: 'high' ({high}) must be > 'low' ({low})")

    # Scale parameters must be positive (normal, laplace, exponential, exp_normal)
    if "scale" in params:
        try:
            scale_val = float(params["scale"])
        except (ValueError, TypeError) as e:
            raise ValueError(f"'scale' must be numeric, got {params['scale']}") from e

        if scale_val <= 0:
            raise ValueError(f"'scale' must be > 0, got {scale_val}")

    # Sigma must be positive (lognormal)
    if "sigma" in params:
        try:
            sigma_val = float(params["sigma"])
        except (ValueError, TypeError) as e:
            raise ValueError(f"'sigma' must be numeric, got {params['sigma']}") from e

        if sigma_val <= 0:
            raise ValueError(f"'sigma' must be > 0, got {sigma_val}")

    # loc and mean must be numeric if present
    for param_name in ["loc", "mean"]:
        if param_name in params:
            try:
                float(params[param_name])
            except (ValueError, TypeError) as e:
                raise ValueError(f"'{param_name}' must be numeric, got {params[param_name]}") from e

    return params


class ClassConfig(BaseModel):
    """Configuration for a single class in the dataset.

    Each class is defined by its sample count, distribution, and optional label.
    Class indices (0, 1, 2, ...) are assigned by position in the list.
    Auto-generated labels follow pattern “class_{idx}”.

    Args:
        n_samples: Number of samples for this class (must be >= 1).
        class_distribution: Distribution type for feature generation. Supported numpy random generator distributions:
            - "normal", "lognormal", "uniform", "exponential", "laplace". Additionally, "exp_normal" for direct control
                over lognormal parameters.
        class_distribution_params: Parameters for the chosen distribution.
        label: Optional descriptive name. Auto-generated as "class_0", "class_1", etc. if not provided.

    Examples:
        >>> # Auto-generated labels
        >>> configs = [
        ...     ClassConfig(n_samples=100),  # label → "class_0"
        ...     ClassConfig(n_samples=50)    # label → "class_1"
        ... ]

        >>> # Explicit semantic labels
        >>> configs = [
        ...     ClassConfig(n_samples=100, label="healthy"),
        ...     ClassConfig(n_samples=50, label="diseased")
        ... ]

        >>> # Different distributions per class
        >>> configs = [
        ...     ClassConfig(
        ...         n_samples=50,
        ...         label="control",
        ...         class_distribution="normal"
        ...     ),
        ...     ClassConfig(
        ...         n_samples=30,
        ...         label="diseased",
        ...         class_distribution="lognormal",
        ...         class_distribution_params={"mean": 0, "sigma": 0.5}
        ...     )
        ... ]
    """

    model_config = ConfigDict(extra="forbid")

    n_samples: int = Field(
        default=30,
        ge=1,
        description="Number of samples in this class.",
    )
    class_distribution: DistributionType = Field(
        default="normal",
        description="Distribution type for base feature generation.",
    )
    class_distribution_params: dict[str, Any] | None = Field(
        default=None,
        description="Distribution parameters. If None, distribution-specific defaults are derived.",
    )
    label: str | None = Field(
        default=None,
        description="Label (auto-generated as 'class_{idx}' if None).",
    )

    @field_validator("class_distribution_params")
    @classmethod
    def _validate_class_params(cls, v: dict[str, float] | None, info) -> dict[str, float] | None:
        """Validate distribution parameters match the chosen distribution."""
        if v is None:
            return v
        distribution = info.data.get("class_distribution", "normal")
        return validate_distribution_params(v, distribution)

    @model_validator(mode="after")
    def _resolve_distribution_params(self):
        """Derive distribution-specific default parameters when none are given.

        Keeping resolution at config time means the constructed config is always
        complete and internally consistent: the parameters can never disagree
        with the chosen distribution.
        """
        if self.class_distribution_params is None:
            object.__setattr__(
                self,
                "class_distribution_params",
                default_distribution_params(self.class_distribution),
            )
        return self

    def __str__(self) -> str:
        """Concise string representation."""
        parts = [f"n={self.n_samples}"]
        if self.label:
            parts.append(f"label='{self.label}'")
        if self.class_distribution != "normal":
            parts.append(f"dist={self.class_distribution}")
        return f"ClassConfig({', '.join(parts)})"


class BatchEffectsConfig(BaseModel):
    """Configuration for simulating batch effects.

    Simulate batch effects by adding random intercepts or scaling factors
    to a subset of features. This can be used to mimic:
      - site-to-site differences (multi-center studies),
      - instrument calibration shifts,
      - cohort / recruitment waves (temporal batches).

    **Conceptual separation of batch effect aspects**:
        - ``confounding_with_class`` controls **sampling bias**:
          which samples (classes) are recruited into which batch.

        - ``effect_strength``, ``effect_type`` and ``effect_granularity`` control
          **technical variation**: how strongly, and how coherently across features,
          the measurements shift between batches.

    Args:
        n_batches:
            Number of batches. Value 0 effectively disables batch effects.
        effect_strength: Scale of batch effects. Must be non-negative.
            - For ``effect_type="additive"``: standard deviation of the additive
                batch effects, sampled as ``Normal(0, effect_strength)``.
            - For ``effect_type="multiplicative"``: standard deviation of the
                  multiplicative deviations around 1.0, sampled as
                  ``1 + Normal(0, effect_strength)``.
        effect_type: Type of batch effect.
            - ``"additive"``: Additive intercepts (shifts in feature means).
            - ``"multiplicative"``: Multiplicative scaling (changes in variance/scale).
        effect_granularity: Granularity of batch effects across features:
            - ``"per_feature"``: draw distinct effects per batch and affected
                    feature (shape ``(n_batches, n_affected_features)``).
            - ``"scalar"``: draw a single effect per batch and apply it
                uniformly to all affected features (global per-batch shift/scale).
        confounding_with_class: Degree of confounding between batch and class in ``[0.0, 1.0]``.
            Controls how strongly batch assignment correlates with class labels,
            simulating **recruitment bias** in multi-center studies.

            Semantics (for two classes / two batches with equal base proportions):
                - 0.0 → independent: each batch has ~50/50 class mix.
                - 0.5 → moderate correlation.
                - 0.8 → strong recruitment bias (most samples of a class go to
                  one batch).
                - 1.0 → perfect confounding: each class maps to one preferred
                  batch (if ``n_batches >= n_classes``).
        affected_features: Which features should be affected:
            - ``"all"``: apply batch effects to all features.
            - list of ints: explicit 0-based column indices of affected features.
        proportions: Optional target proportions for batch sizes. Values are normalized
            to sum to 1. If ``None``, batches are (approximately) equal in size.
    """

    model_config = ConfigDict(extra="forbid")

    # 0 or 1 => effectively no batch effect
    n_batches: int = Field(default=0, ge=0)

    # std of batch effects (0.0 allowed => no effect)
    effect_strength: float = Field(default=0.5, ge=0.0)

    effect_type: Literal["additive", "multiplicative"] = "additive"

    # how structured across features: per-feature vs scalar per batch
    effect_granularity: Literal["per_feature", "scalar"] = Field(default="per_feature")

    # in [0, 1], controls recruitment bias / confounding
    confounding_with_class: float = Field(default=0.0, ge=0.0, le=1.0)

    # 0-based column indices; "all" => all features
    affected_features: list[int] | Literal["all"] = "all"

    # optional batch size proportions
    proportions: list[float] | None = None

    @field_validator("proportions")
    @classmethod
    def validate_proportions(cls, v: list[float] | None, info):
        """Ensure proportions are non-negative, match n_batches, and sum to 1."""
        if v is None:
            return v

        if len(v) == 0:
            raise ValueError("proportions must not be empty if provided.")

        # Non-negative entries
        for p in v:
            if p < 0:
                raise ValueError(f"proportions must be non-negative, got {p}.")

        # Check length vs n_batches (if > 0)
        n_batches = info.data.get("n_batches")
        if isinstance(n_batches, int) and 0 < n_batches != len(v):
            raise ValueError(f"proportions length ({len(v)}) must match n_batches ({n_batches}).")

        total = float(sum(v))
        if total <= 0:
            raise ValueError(f"Sum of proportions must be > 0, got {total}.")

        # Normalize to sum to 1.0
        return [p / total for p in v]


def _correlation_in_range(correlation: float, n_features: int, structure: str) -> bool:
    """Return whether a correlation is admissible for a block of the given shape.

    Args:
        correlation: Candidate within-block correlation.
        n_features: Number of features in the block.
        structure: Either ``"equicorrelated"`` or ``"toeplitz"``.

    Returns:
        True if the correlation yields a valid (positive-definite) block.
    """
    if structure == "equicorrelated":
        lower = -1.0 / (n_features - 1)
        return lower < correlation < 1.0
    return -1.0 < correlation < 1.0


class MeanChannel(BaseModel):
    """First-moment signal: a per-class mean shift applied to the cluster anchor.

    Absent classes receive a 0.0 shift (baseline). Shifts are in standard-
    deviation units of the standardized feature baseline.

    Attributes:
        per_class_effect: Mapping from class index to mean shift in sigma units.
    """

    model_config = ConfigDict(extra="forbid")

    per_class_effect: dict[int, float]


class CovarianceChannel(BaseModel):
    """Second-moment signal: a per-class within-cluster correlation.

    Absent classes fall back to the cluster's ``baseline_correlation``. This
    models differential co-expression.

    Attributes:
        per_class_correlation: Mapping from class index to within-cluster correlation.
    """

    model_config = ConfigDict(extra="forbid")

    per_class_correlation: dict[int, float]


class CorrClusterConfig(BaseModel):
    """A correlated block with optional, independent mean and covariance channels.

    The block geometry and the structural anchor are always present; signal is
    expressed only through the optional channels. Relevance is derived (a cluster
    is informative iff a channel varies across classes), never declared.

    Anchor-to-proxy mean propagation is not configured directly: a proxy at block
    column ``j`` inherits ``effect * proxy_attenuation * structural_correlation[anchor_index, j]``,
    where the structural correlation matrix is built from ``correlation_structure``
    and the effective per-class correlation (the covariance channel value for that
    class, or ``baseline_correlation`` when absent). With the default
    ``proxy_attenuation=1.0`` this reproduces the v1 propagation model exactly, and
    uses the same correlation that samples the block.

    Attributes:
        n_cluster_features: Number of features in the block (>= 2): one anchor plus proxies.
        correlation_structure: Within-block correlation pattern.
        baseline_correlation: Structural correlation used when no covariance channel
            overrides a given class. ``0.0`` means independence.
        anchor_index: Index (within the block) of the structural anchor.
        proxy_attenuation: Neutral multiplier on the structurally derived anchor-to-proxy
            mean propagation. ``1.0`` reproduces the v1 model (no extra attenuation).
        mean_channel: Optional first-moment signal.
        covariance_channel: Optional second-moment signal.
        label: Optional descriptive name for documentation.
    """

    model_config = ConfigDict(extra="forbid")

    n_cluster_features: int = Field(..., ge=2)
    correlation_structure: Literal["equicorrelated", "toeplitz"] = "equicorrelated"
    baseline_correlation: float = 0.0
    anchor_index: int = 0
    proxy_attenuation: float = 1.0
    mean_channel: MeanChannel | None = None
    covariance_channel: CovarianceChannel | None = None
    label: str | None = None

    @model_validator(mode="after")
    def _validate_structure(self):
        """Validate anchor index and all correlations against the block shape."""
        if not (0 <= self.anchor_index < self.n_cluster_features):
            raise ValueError(f"anchor_index must be in [0, {self.n_cluster_features}), got {self.anchor_index}.")

        if not _correlation_in_range(self.baseline_correlation, self.n_cluster_features, self.correlation_structure):
            raise ValueError(
                f"baseline_correlation={self.baseline_correlation} invalid for "
                f"{self.correlation_structure} with n_cluster_features={self.n_cluster_features}."
            )

        if self.covariance_channel is not None:
            for class_index, rho in self.covariance_channel.per_class_correlation.items():
                if not _correlation_in_range(float(rho), self.n_cluster_features, self.correlation_structure):
                    raise ValueError(
                        f"covariance_channel correlation for class {class_index} is {rho}, "
                        f"invalid for {self.correlation_structure} with "
                        f"n_cluster_features={self.n_cluster_features}."
                    )
        return self

    # Channel resolution -------------------------------------------------------

    def effective_correlation_for_class(self, class_index: int) -> float:
        """Resolve the within-block correlation for a class.

        The covariance channel value for ``class_index`` if present, otherwise the
        cluster's ``baseline_correlation``.
        """
        if self.covariance_channel is None:
            return float(self.baseline_correlation)
        return float(self.covariance_channel.per_class_correlation.get(class_index, self.baseline_correlation))

    def mean_effect_for_class(self, class_index: int) -> float:
        """Resolve the anchor mean shift for a class (0.0 when absent or no channel)."""
        if self.mean_channel is None:
            return 0.0
        return float(self.mean_channel.per_class_effect.get(class_index, 0.0))


class StandaloneInformativeGroup(BaseModel):
    """A group of standalone informative features sharing one separation strength.

    All members are independent (cluster-free) informative features. They share
    the per-class base distribution defined globally on ``ClassConfig`` and differ
    only in the magnitude of the class-wise mean offset, set by ``class_sep``.
    A list of groups with decreasing ``class_sep`` realizes a signal-strength
    gradient across the standalone-informative block.

    Attributes:
        n_features: Number of standalone informative features in this group (>= 1).
        class_sep: Per-class separation. A scalar broadcasts to a length
            ``n_classes - 1`` vector of equal pairwise separations; a sequence
            gives the pairwise separations directly. Offsets are formed exactly as
            for the existing standalone mechanism (centered cumulative sums).
    """

    model_config = ConfigDict(extra="forbid")

    n_features: int = Field(..., ge=1, description="Number of standalone informative features in this group.")
    class_sep: float | Sequence[float] = Field(
        ...,
        description="Per-class separation (scalar broadcast, or sequence of length n_classes - 1).",
    )

    @field_validator("class_sep")
    @classmethod
    def _validate_class_sep_finite(cls, v: float | Sequence[float]) -> float | Sequence[float]:
        """Validate finiteness only; the length-vs-n_classes check lives on DatasetConfig.

        A scalar ``class_sep`` must be finite; every entry of a sequence
        ``class_sep`` must be finite. The length validation against
        ``n_classes - 1`` requires ``n_classes`` and is therefore deferred to
        ``DatasetConfig``.
        """
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if not np.isfinite(float(v)):
                raise ValueError(f"class_sep must be finite, got {v!r}.")
            return v
        if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
            for entry in v:
                try:
                    fentry = float(entry)
                except (TypeError, ValueError) as e:
                    raise TypeError(f"class_sep entries must be numeric, got {entry!r}.") from e
                if not np.isfinite(fentry):
                    raise ValueError(f"class_sep entries must be finite, got {entry!r}.")
            return v
        raise TypeError(f"class_sep must be a number or sequence of numbers, got {type(v).__name__}.")


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
class DatasetConfig(BaseModel):
    """Configuration for synthetic dataset generation.

    This model defines the *input-level* controls for building a synthetic dataset.
    Signal is expressed structurally through channels, never through declared
    relevance. The inputs are:

      - Standalone informative block: ``standalone_informative_groups`` -- a list of
        :class:`StandaloneInformativeGroup`, each contributing independent
        informative features that share one separation strength (``class_sep``).
      - Standalone noise block: ``n_standalone_noise`` independent noise features.
      - Correlated clusters: ``corr_clusters`` -- a list of
        :class:`CorrClusterConfig`. Each cluster is a correlated block (anchor plus
        proxies) whose signal is carried by optional channels: a ``mean_channel``
        (per-class anchor mean shift, first moment), a ``covariance_channel``
        (per-class within-cluster correlation, second moment), and a structural
        ``baseline_correlation`` used when no covariance channel overrides a class.
      - Class definitions: ``class_configs`` -- per-class sample counts, base
        distributions, and labels.
      - Optional batch effects: ``batch_effects``.
      - Reproducibility: ``random_state``.

    *Derived properties* (computed from the inputs above, never set by the user):

        - ``n_samples`` (int): Total samples (from ``class_configs``).
        - ``n_classes`` (int): Number of classes (from ``class_configs``).
        - ``n_features`` (int): Standalone informative + standalone noise +
          cluster members.
        - ``n_standalone_informative`` (int): Sum of ``n_features`` over all
          ``standalone_informative_groups``.
        - ``n_informative`` (int): Standalone informative features plus all members
          of clusters that the signal predicate derives as informative (a cluster
          is informative iff its mean channel varies across classes or its
          effective per-class correlation varies across classes).
        - ``n_noise`` (int): Complement of ``n_informative``.

        Setting any of ``n_samples``, ``n_classes``, ``n_features``,
        ``n_informative``, ``n_noise``, or ``n_standalone_informative`` manually is
        rejected.

    Args:
        standalone_informative_groups (list[StandaloneInformativeGroup]): Groups of
            standalone informative features, each with its own ``class_sep``.
        n_standalone_noise (int): Number of standalone (cluster-free) noise features.
        class_configs (list[ClassConfig]): List of class definitions (>= 2).
        corr_clusters (list[CorrClusterConfig]): Correlated feature clusters with
            optional mean/covariance channels and a structural baseline correlation.
        noise_distribution (str): Distribution for noise features. Any supported
            ``DistributionType``.
        noise_distribution_params (dict): Parameters for the noise distribution.
        prefixed_feature_naming (bool):
            If True, role-based prefixed feature names:
                * Standalone informative: i1, i2, ...
                * Standalone noise:       n1, n2, ...
                * Correlated:             corr{cid}_anchor, corr{cid}_2, ..., corr{cid}_k
            If False, use generic feature_{i} naming. Default: True.
        prefix_informative (str): Prefix for informative features (if prefixed_feature_naming=True). Default: "i".
        prefix_noise (str): Prefix for noise features (if prefixed_feature_naming=True). Default: "n".
        prefix_corr (str): Prefix for correlated cluster features (if prefixed_feature_naming=True). Default: "corr".
        batch_effects (BatchEffectsConfig): Optional BatchEffectsConfig for simulating batch effects.
        random_state (int | None): Global random seed for dataset generation.

    Methods:
        breakdown(): Return dict with derived feature counts (standalone/cluster
            members and the derived informative/noise totals).
        cluster_informative_flags(): Per-cluster booleans for derived informativeness.
        from_yaml(path): Load and validate a config from a YAML file.

    Validation:
        Before model construction:
            - Forbid manual ``n_samples``, ``n_classes``, ``n_features``,
              ``n_informative``, ``n_noise``, ``n_standalone_informative``.
            - Require at least two classes.
        After model construction:
            - Validate sequence ``class_sep`` lengths on each group against
              ``n_classes - 1``.
            - Validate that every per-class channel key (mean and covariance)
              is a valid class index in ``range(n_classes)``.
            - Auto-generate missing class labels as ``class_{idx}``.

    Raises:
        ValueError: On invalid numeric ranges or inconsistent counts.
        TypeError: For invalid types in ``class_configs`` or ``class_sep``.

    Examples:
        >>> from biomedical_data_generator.config import (
        ...     ClassConfig,
        ...     CorrClusterConfig,
        ...     DatasetConfig,
        ...     MeanChannel,
        ...     StandaloneInformativeGroup,
        ... )
        >>> cfg = DatasetConfig(
        ...     standalone_informative_groups=[
        ...         StandaloneInformativeGroup(n_features=5, class_sep=1.0),
        ...     ],
        ...     n_standalone_noise=3,
        ...     class_configs=[
        ...         ClassConfig(n_samples=50, label="healthy"),
        ...         ClassConfig(n_samples=50, label="diseased"),
        ...     ],
        ...     corr_clusters=[
        ...         CorrClusterConfig(
        ...             n_cluster_features=4,
        ...             baseline_correlation=0.8,
        ...             mean_channel=MeanChannel(per_class_effect={1: 1.5}),
        ...             label="Metabolic Pathway A",
        ...         ),
        ...         CorrClusterConfig(
        ...             n_cluster_features=3,
        ...             baseline_correlation=0.5,
        ...             label="Structural Correlated Block",
        ...         ),
        ...     ],
        ...     noise_distribution="normal",
        ...     noise_distribution_params={"loc": 0, "scale": 1},
        ...     prefixed_feature_naming=True,
        ...     random_state=42,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    # Core dataset structure (standalone, non-cluster features)
    standalone_informative_groups: list[StandaloneInformativeGroup] = Field(
        default_factory=list,
        description="Groups of standalone informative features, each with its own separation strength.",
    )
    n_standalone_noise: int = Field(default=0, ge=0)

    # Multi-class controls
    class_configs: list[ClassConfig] = Field(
        [ClassConfig(n_samples=30, label="healthy"), ClassConfig(n_samples=30, label="diseased")], min_length=2
    )

    # Noise distribution (NumPy Generator API)
    noise_distribution: DistributionType = "normal"
    noise_distribution_params: dict[str, Any] | None = Field(
        default=None,
        description="Parameters for the noise distribution. If None, distribution-specific defaults are derived.",
    )

    # Naming
    prefixed_feature_naming: bool = True
    prefix_informative: str = "i"
    prefix_noise: str = "n"

    prefix_corr: str = "corr"

    # Correlated structure
    corr_clusters: list[CorrClusterConfig] = Field(default_factory=list)

    # Batch effects
    batch_effects: BatchEffectsConfig | None = None

    # Global seed
    random_state: int | None = None

    # ------------------------------------------------------ before validator

    @model_validator(mode="before")
    @classmethod
    def _normalize_and_validate(cls, data: Any) -> Any:
        """Validate incoming data BEFORE model construction.

        Forbids manual setting of derived counts and requires at least two
        classes. Per-class separation now lives on each
        ``StandaloneInformativeGroup`` (``class_sep``); there is no top-level
        ``class_sep`` to normalize here.
        """
        if isinstance(data, cls):
            return data

        if not isinstance(data, Mapping):
            raise TypeError(f"DatasetConfig expects a mapping-like raw_config, got {type(data).__name__}")

        d: dict[str, Any] = dict(data)

        # Forbid manual override of derived attributes
        for forbidden in (
            "n_samples",
            "n_classes",
            "n_features",
            "n_informative",
            "n_noise",
            "n_standalone_informative",
        ):
            if forbidden in d:
                raise ValueError(
                    f"{forbidden} is derived from class_configs/corr_clusters/"
                    "standalone_informative_groups and must not be set manually on DatasetConfig."
                )

        classes = d.get("class_configs")
        if not isinstance(classes, Sequence) or isinstance(classes, (str, bytes)):
            raise TypeError("class_configs must be a non-string sequence of class definitions.")

        n_classes = len(classes)
        if n_classes < 2:
            raise ValueError(f"At least two classes are required, got {n_classes}.")

        return d

    # ------------------------------------------------------ field validation

    @field_validator("noise_distribution_params")
    @classmethod
    def _validate_noise_params(cls, v: dict[str, float] | None, info) -> dict[str, float] | None:
        """Validate distribution parameters match the chosen noise distribution."""
        if v is None:
            return v
        distribution = info.data.get("noise_distribution", "normal")
        return validate_distribution_params(v, distribution)

    @model_validator(mode="after")
    def _resolve_noise_distribution_params(self):
        """Derive distribution-specific default noise parameters when none are given.

        Without this, selecting a non-default ``noise_distribution`` (e.g.
        "uniform") and omitting parameters would either fail validation or leave
        the config carrying parameters that do not match the distribution.
        """
        if self.noise_distribution_params is None:
            object.__setattr__(
                self,
                "noise_distribution_params",
                default_distribution_params(self.noise_distribution),
            )
        return self

    # ------------------------------------------------------ after validators
    @model_validator(mode="after")
    def _auto_generate_labels(self):
        """Auto-generate labels as 'class_{idx}' if not provided."""
        for idx, cls_cfg in enumerate(self.class_configs):
            if cls_cfg.label is None or cls_cfg.label == "":
                # ClassConfig is a BaseModel, so we need object.__setattr__
                object.__setattr__(cls_cfg, "label", f"class_{idx}")
        return self

    @model_validator(mode="after")
    def _validate_group_class_sep_lengths(self):
        """Validate sequence ``class_sep`` lengths on standalone informative groups.

        A scalar ``class_sep`` is always valid (it broadcasts). A sequence must
        have length ``n_classes - 1``. Finiteness is already enforced on
        ``StandaloneInformativeGroup``; only the length check needs ``n_classes``
        and therefore lives here.

        Raises:
            ValueError: If a group's sequence ``class_sep`` has the wrong length.
        """
        expected = self.n_classes - 1
        for group_id, group in enumerate(self.standalone_informative_groups):
            sep = group.class_sep
            if isinstance(sep, Sequence) and not isinstance(sep, (str, bytes)):
                if len(sep) != expected:
                    raise ValueError(
                        f"standalone_informative_groups[{group_id}].class_sep has length "
                        f"{len(sep)}, but must be n_classes - 1 ({expected})."
                    )
        return self

    @model_validator(mode="after")
    def _validate_channel_class_keys(self):
        """Validate that every per-class channel key is a valid class index.

        The channels are the only place class indices are referenced now; their
        keys must lie in ``range(n_classes)``. This is the cross-cutting check
        that needs ``n_classes`` and therefore lives on ``DatasetConfig`` rather
        than on the channel models.

        Raises:
            ValueError: If a mean- or covariance-channel key is out of range.
        """
        n_classes = self.n_classes
        for cluster_id, cluster in enumerate(self.corr_clusters or []):
            if cluster.mean_channel is not None:
                for class_index in cluster.mean_channel.per_class_effect:
                    if not (0 <= class_index < n_classes):
                        raise ValueError(
                            f"corr_clusters[{cluster_id}].mean_channel has class index "
                            f"{class_index}, but only {n_classes} classes are defined."
                        )
            if cluster.covariance_channel is not None:
                for class_index in cluster.covariance_channel.per_class_correlation:
                    if not (0 <= class_index < n_classes):
                        raise ValueError(
                            f"corr_clusters[{cluster_id}].covariance_channel has class index "
                            f"{class_index}, but only {n_classes} classes are defined."
                        )
        return self

    @classmethod
    def from_yaml(cls, path: str) -> DatasetConfig:
        """Load from YAML and validate via the same pipeline."""
        import yaml  # local import to keep core dependencies lean

        with open(path, encoding="utf-8") as f:
            raw_config: dict[str, Any] = yaml.safe_load(f) or {}
        return cls.model_validate(raw_config)

    def cluster_informative_flags(self) -> list[bool]:
        """Per-cluster booleans: whether each cluster is *derived* informative.

        Relevance is derived from the channel mappings via the shared predicate,
        never declared. A cluster is informative iff its mean channel varies
        across classes or its effective per-class correlation varies across
        classes.

        Returns:
            One boolean per cluster, in ``corr_clusters`` order.
        """
        flags: list[bool] = []
        for cluster in self.corr_clusters or []:
            mean_per_class = cluster.mean_channel.per_class_effect if cluster.mean_channel is not None else None
            covariance_per_class = (
                cluster.covariance_channel.per_class_correlation if cluster.covariance_channel is not None else None
            )
            flags.append(
                _cluster_is_informative(
                    mean_per_class=mean_per_class,
                    covariance_per_class=covariance_per_class,
                    baseline_correlation=cluster.baseline_correlation,
                    n_classes=self.n_classes,
                )
            )
        return flags

    def cluster_column_informative_flags(self) -> list[list[bool]]:
        """Per-column booleans: whether each cluster column is *derived* informative.

        The per-column refinement of :meth:`cluster_informative_flags`. Relevance
        is derived per column from the channel mappings via the shared predicate,
        never declared. A cluster column is informative iff it carries a
        class-dependent mean shift (the anchor's shift, or a proxy's attenuated
        propagation) or participates in a class-dependent within-cluster
        correlation. A single cluster may therefore split across informative and
        noise roles (e.g. an informative anchor with noise proxies).

        Returns:
            One inner list per cluster, in ``corr_clusters`` order, with one
            boolean per column (anchor and proxies) in block-column order.
        """
        flags: list[list[bool]] = []
        for cluster in self.corr_clusters or []:
            mean_per_class = cluster.mean_channel.per_class_effect if cluster.mean_channel is not None else None
            covariance_per_class = (
                cluster.covariance_channel.per_class_correlation if cluster.covariance_channel is not None else None
            )
            cluster_flags = [
                _cluster_column_carries_signal(
                    mean_per_class=mean_per_class,
                    covariance_per_class=covariance_per_class,
                    baseline_correlation=cluster.baseline_correlation,
                    correlation_structure=cluster.correlation_structure,
                    proxy_attenuation=cluster.proxy_attenuation,
                    distance=abs(position - cluster.anchor_index),
                    n_classes=self.n_classes,
                )
                for position in range(cluster.n_cluster_features)
            ]
            flags.append(cluster_flags)
        return flags

    @property
    def n_standalone_informative(self) -> int:
        """Derived count of standalone (cluster-free) informative features.

        Sum of ``n_features`` across all :attr:`standalone_informative_groups`.
        """
        return int(sum(group.n_features for group in self.standalone_informative_groups))

    @property
    def n_informative(self) -> int:
        """Derived informative feature count.

        Standalone informative features plus all members of clusters that the
        signal predicate marks informative.
        """
        cluster_informative = sum(sum(column_flags) for column_flags in self.cluster_column_informative_flags())
        return int(self.n_standalone_informative + cluster_informative)

    @property
    def n_noise(self) -> int:
        """Derived noise feature count: the complement of :attr:`n_informative`."""
        return int(self.n_features - self.n_informative)

    # ------------------------------ derived global counts ---------------------

    @property
    def n_samples(self) -> int:
        """Total samples (derived from class_configs)."""
        return sum(c.n_samples for c in self.class_configs)

    @property
    def n_classes(self) -> int:
        """Number of classes (derived from class_configs)."""
        return len(self.class_configs)

    @property
    def n_features(self) -> int:
        """Total number of features: standalone informative + standalone noise + cluster members."""
        cluster_members = sum(int(c.n_cluster_features) for c in (self.corr_clusters or []))
        return int(self.n_standalone_informative + self.n_standalone_noise + cluster_members)

    # ------------------------------ class-level helpers ----------------------

    @property
    def class_labels(self) -> list[str]:
        """List of class labels (auto-generated or user-provided)."""
        return [
            c.label if (c.label is not None and c.label != "") else f"class_{i}"
            for i, c in enumerate(self.class_configs)
        ]

    @property
    def class_counts(self) -> dict[int, int]:
        """Class counts as dict {class_idx: n_samples}."""
        return {idx: c.n_samples for idx, c in enumerate(self.class_configs)}

    def breakdown(self) -> dict[str, int]:
        """Structured, derived feature counts.

        Returns:
            A dict with keys:
            - n_standalone_informative
            - n_standalone_noise
            - n_cluster_members
            - n_informative (derived)
            - n_noise (derived)
            - n_features
        """
        cluster_members = sum(int(c.n_cluster_features) for c in (self.corr_clusters or []))
        return {
            "n_standalone_informative": int(self.n_standalone_informative),
            "n_standalone_noise": int(self.n_standalone_noise),
            "n_cluster_members": int(cluster_members),
            "n_informative": int(self.n_informative),
            "n_noise": int(self.n_noise),
            "n_features": int(self.n_features),
        }
