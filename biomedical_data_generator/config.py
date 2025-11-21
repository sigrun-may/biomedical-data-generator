# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Configuration models for the dataset generator."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AnchorMode: TypeAlias = Literal["equalized", "strong"]
DistributionType = Literal[
    "normal",
    "lognormal",
    "exp_normal",  # np.exp(rng.normal()) - direct control over underlying parameters for lognormal distribution
    "uniform",
    "exponential",
    "laplace",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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

    param_schema = {
        "normal": {"required": set(), "optional": {"loc", "scale"}},
        "uniform": {"required": {"low", "high"}, "optional": set()},
        "laplace": {"required": set(), "optional": {"loc", "scale"}},
        "exponential": {"required": set(), "optional": {"scale"}},
        "lognormal": {"required": set(), "optional": {"mean", "sigma"}},
        "exp_normal": {"required": set(), "optional": {"loc", "scale"}},
    }

    schema = param_schema.get(distribution)
    if not schema:
        return params

    provided = set(params.keys())
    invalid = provided - (schema["required"] | schema["optional"])
    if invalid:
        raise ValueError(
            f"Invalid parameters {invalid} for '{distribution}'. " f"Allowed: {schema['required'] | schema['optional']}"
        )

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


# ---------------------------------------------------------------------------
# Class configuration
# ---------------------------------------------------------------------------


class ClassConfig(BaseModel):
    """Configuration for a single class in the dataset.

    Each class is defined by its sample count, distribution, and optional label.
    Class labels (0, 1, 2, ...) are assigned by position in the list.

    Args:
        n_samples: Number of samples for this class (must be >= 1).
        class_distribution: Distribution type for feature generation.
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
        ...     ClassConfig(n_samples=50, label="control", class_distribution="normal"),
        ...     ClassConfig(
        ...         n_samples=30,
        ...         label="diseased",
        ...         class_distribution="lognormal",
        ...         class_distribution_params={"mean": 0, "sigma": 0.5}
        ...     )
        ... ]

    Notes:
        - Class index is determined by position: first config = class 0, etc.
        - n_samples is exact (not a weight or proportion)
        - label is auto-generated if None or empty string
        - Auto-generated labels follow pattern "class_{idx}"
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
    class_distribution_params: dict[str, Any] = Field(
        default_factory=lambda: {"loc": 0, "scale": 1},
        description="Distribution parameters.",
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
    to specified features. This can be used to mimic:
      - site-to-site differences (multi-center studies),
      - instrument calibration shifts,
      - cohort / recruitment waves (temporal batches).

    Randomness and reproducibility
    ------------------------------
    Batch effects are integrated into the dataset-level RNG in a predictable way:

    - In :func:`generate_dataset`, a single
      ``rng_global = np.random.default_rng(cfg.random_state)`` is created.

    - If ``BatchEffectsConfig.random_state`` is ``None`` (recommended),
      **all batch-related draws reuse this global generator**:
        * batch assignment (random or confounded),
        * batch effects (additive or multiplicative).
      In this default mode, ``DatasetConfig.random_state`` is the **single knob**
      that reproduces the entire dataset, *including* batch effects.

    - If ``BatchEffectsConfig.random_state`` is an integer, a dedicated RNG
      ``np.random.default_rng(batch.random_state)`` is created and passed to the
      batch-effect routines. Use this only if you explicitly want to keep batch
      effects fixed while changing other aspects of the dataset.

    Conceptual separation
    ---------------------
    - ``confounding_with_class`` controls **sampling bias**:
      which samples (classes) are recruited into which batch.

    - ``effect_strength`` and ``effect_type`` control **technical variation**:
      how strongly the measurement shifts between batches.

    Args:
        n_batches:
            Number of batches. Values 0 or 1 effectively disable batch effects.
        effect_strength:
            Scale of batch effects.
            - For ``effect_type="additive"``: standard deviation of batch intercepts.
            - For ``effect_type="multipliclicative"``: standard deviation of the
              multiplicative factor (applied as ``X' = X * (1 + b_batch)``).
        effect_type:
            Type of batch effect.
            - ``"additive"``: Additive intercepts (shifts in feature means).
            - ``"multiplicative"``: Multiplicative scaling (changes in variance/scale).
        confounding_with_class: Degree of confounding between batch and class (0.0-1.0).
            Controls how strongly batch assignment correlates with class labels,
            simulating **recruitment bias** in multi-center studies.

            The parameter determines the probability that samples from the same
            class are assigned to the same batch:
            Semantics (for two classes / two batches, equal base proportions):
                - 0.0 → independent: each batch has ~50/50 class mix.
                - 0.5 → moderate correlation
                - 0.8 → strong recruitment bias (most samples of a class go to one batch).
                - 1.0 → perfect confounding: each class maps to one preferred batch
                  (if ``n_batches >= n_classes``).
        affected_features (list[int] | Literal["all", "informative"]):
            Which features should be affected:
            - ``"all"``: apply batch effects to all features.
            - ``"informative"``: apply only to informative features.
            - list of ints: explicit 0-based column indices.
        proportions:
            Optional target proportions for batch sizes. Values are normalized
            to sum to 1. If ``None``, batches are (approximately) equal in size.
        random_state:
            Optional seed specific to batch effects. ``None`` → reuse dataset RNG.
    """

    model_config = ConfigDict(extra="forbid")

    n_batches: int = Field(default=0, ge=0)  # 0 or 1 => no batch effect
    effect_strength: float = Field(default=0.5, gt=0.0)  # std of batch effects
    effect_type: Literal["additive", "multiplicative"] = "additive"
    confounding_with_class: float = Field(default=0.0, ge=0.0, le=1.0)  # in [0,1]
    affected_features: list[int] | Literal["all", "informative"] = "all"  # 0-based column indices; "all" => all
    proportions: list[float] | None = None  # optional batch proportions
    random_state: int | None = None

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
        if isinstance(n_batches, int) and n_batches > 0 and len(v) != n_batches:
            raise ValueError(f"proportions length ({len(v)}) must match n_batches ({n_batches}).")

        total = float(sum(v))
        if total <= 0:
            raise ValueError(f"Sum of proportions must be > 0, got {total}.")

        # Normalize to sum to 1.0
        return [p / total for p in v]


# ---------------------------------------------------------------------------
# Correlated clusters
# ---------------------------------------------------------------------------


class CorrClusterConfig(BaseModel):
    """Correlated feature cluster simulating coordinated biomarker patterns.

    A cluster represents a group of biomarkers that move together, such as
    markers in a metabolic pathway or proteins in a signaling cascade. One marker
    acts as the "anchor" (driver), while the others are "proxies" (followers).

    Args:
        n_cluster_features:
            Number of biomarkers in the cluster (including anchor). Must be >= 1.
        rho:
            Correlation strength between biomarkers in the cluster:
              - 0.0 = independent
              - 0.5 = moderate correlation
              - 0.8+ = strong correlation
            For equicorrelated:
                - lower bound = -1 / (p-1) for p > 1
                - upper bound < 1
            For toeplitz:
                - |rho| < 1
        structure:
            "equicorrelated" or "toeplitz".
        class_structure:
            Per-class override for correlation structure.
        class_rho:
            Per-class override for correlation strength.
        rho_baseline:
            Fallback correlation for classes not in class_rho (default 0.0).
        anchor_role:
            "informative" or "noise".
        anchor_effect_size:
            "small" (0.5), "medium" (1.0), "large" (1.5), custom > 0, or None.
        anchor_class:
            Class index the anchor predicts (if informative). None → all classes.
        random_state:
            Optional seed for this cluster. If None, uses the global dataset seed.
        label:
            Descriptive name for documentation.
    """

    model_config = ConfigDict(extra="forbid")

    # Core cluster structure and correlation settings
    n_cluster_features: int = Field(
        ...,
        ge=1,
        description="Number of biomarkers in cluster (including anchor).",
    )

    structure: Literal["equicorrelated", "toeplitz"] = Field(
        default="equicorrelated",
        description="Default correlation structure for all classes.",
    )
    rho: float = Field(
        default=0.8,
        description="Default correlation strength for all classes.",
    )

    # Per-class correlation (optional)
    class_structure: dict[int, Literal["equicorrelated", "toeplitz"]] | None = Field(
        default=None,
        description="Per-class correlation structure (overrides 'structure' for specified classes).",
    )
    class_rho: dict[int, float] | None = Field(
        default=None,
        description="Per-class correlation strength (activates class-specific mode).",
    )
    rho_baseline: float = Field(
        default=0.0,
        ge=-1.0,
        lt=1.0,
        description="Fallback correlation for classes not in class_rho.",
    )

    # Biological relevance
    anchor_role: Literal["informative", "noise"] = "informative"
    anchor_effect_size: Literal["small", "medium", "large"] | float | None = None
    anchor_class: int | None = None

    # Metadata
    random_state: int | None = None
    label: str | None = None

    @field_validator("n_cluster_features")
    @classmethod
    def _validate_size(cls, v: int) -> int:
        """Ensure cluster has at least one marker."""
        if v < 1:
            raise ValueError(f"n_cluster_features must be >= 1, got {v}.")
        return v

    @field_validator("rho")
    @classmethod
    def _validate_rho_range(cls, v: float, info):
        p = int(info.data.get("n_cluster_features", 0))
        structure = info.data.get("structure", "equicorrelated")
        if structure == "equicorrelated":
            lower = -1.0 / (p - 1) if p > 1 else float("-inf")
            if not (lower < v < 1.0):
                raise ValueError(
                    f"rho={v} invalid for equicorrelated with n_cluster_features={p}; "
                    f"require {lower:.6f} < rho < 1."
                )
        else:  # toeplitz
            if not (-1.0 < v < 1.0):
                raise ValueError("For toeplitz, require |rho| < 1.")
        return v

    @field_validator("class_rho")
    @classmethod
    def _validate_class_rho(cls, v, info):
        if v is None:
            return v
        p = int(info.data.get("n_cluster_features", 0))
        structure = info.data.get("structure", "equicorrelated")
        lower = -1.0 / (p - 1) if (structure == "equicorrelated" and p > 1) else -1.0
        for cls_idx, rho_val in v.items():
            if cls_idx < 0:
                raise ValueError(f"class_rho keys must be >= 0, got {cls_idx}.")
            if not (lower < float(rho_val) < 1.0):
                raise ValueError(
                    f"class_rho[{cls_idx}]={rho_val} invalid for {structure} (p={p}); "
                    f"require {lower:.6f} < rho < 1."
                )
        return v

    @field_validator("rho_baseline")
    @classmethod
    def _validate_rho_baseline(cls, v: float, info):
        """Validate rho_baseline against structure constraints.

        rho_baseline must satisfy the same positive-definiteness constraints
        as rho and class_rho, since it's used as a correlation parameter
        for classes not specified in class_rho.

        Args:
            v: The rho_baseline value to validate.
            info: Validation context with access to other fields.

        Returns:
            The validated rho_baseline value.

        Raises:
            ValueError: If rho_baseline violates PD constraints for the structure.
        """
        p = int(info.data.get("n_cluster_features", 2))
        structure = info.data.get("structure", "equicorrelated")

        if structure == "equicorrelated":
            lower = -1.0 / (p - 1) if p > 1 else float("-inf")
            if not (lower < v < 1.0):
                raise ValueError(
                    f"rho_baseline={v} invalid for equicorrelated with "
                    f"n_cluster_features={p}; require {lower:.6f} < rho_baseline < 1.0"
                )
        else:  # toeplitz
            if not (-1.0 < v < 1.0):
                raise ValueError(f"rho_baseline={v} invalid for toeplitz; require |rho_baseline| < 1.0")

        return v

    @field_validator("class_structure")
    @classmethod
    def _validate_class_structure(cls, v: dict[int, str] | None) -> dict[int, str] | None:
        """Validate per-class structure keys."""
        if v is None:
            return v

        for cls_idx in v.keys():
            if cls_idx < 0:
                raise ValueError(f"class_structure keys must be >= 0, got {cls_idx}.")
        return v

    @model_validator(mode="after")
    def _validate_class_specific_consistency(self):
        """Ensure class_structure is only used with class_rho."""
        if self.class_structure is not None and self.class_rho is None:
            raise ValueError(
                "class_structure requires class_rho to be set. "
                "Either set class_rho (activates class-specific mode) or remove class_structure."
            )
        return self

    @field_validator("anchor_effect_size")
    @classmethod
    def _validate_effect_size(cls, v) -> Literal["small", "medium", "large"] | float | None:
        """Validate effect size is either a preset or positive float."""
        if v is None:
            return v

        # Preset string
        if isinstance(v, str):
            if v not in ("small", "medium", "large"):
                raise ValueError("anchor_effect_size must be 'small', 'medium', or 'large', " f"got '{v}'.")
            return cast(Literal["small", "medium", "large"], v)

        # Custom float
        try:
            val = float(v)
            if val <= 0:
                raise ValueError(f"anchor_effect_size must be > 0, got {val}.")
            return val
        except (TypeError, ValueError) as e:
            raise ValueError(
                "anchor_effect_size must be 'small'/'medium'/'large' or positive float, " f"got {v}."
            ) from e

    @field_validator("anchor_class")
    @classmethod
    def _validate_anchor_class(cls, v: int | None) -> int | None:
        """Ensure disease class is non-negative if specified."""
        if v is not None and v < 0:
            raise ValueError(f"anchor_class must be >= 0 or None, got {v}.")
        return v

    # Convenience methods -----------------------------------------------------

    def is_class_specific(self) -> bool:
        """Check if this cluster uses class-specific correlation."""
        return self.class_rho is not None

    def get_rho_for_class(self, class_idx: int) -> float:
        """Get correlation strength for a specific class."""
        if not self.is_class_specific():
            return self.rho
        assert self.class_rho is not None
        return self.class_rho.get(class_idx, self.rho_baseline)

    def get_structure_for_class(self, class_idx: int) -> Literal["equicorrelated", "toeplitz"]:
        """Get correlation structure for a specific class."""
        if self.class_structure is None:
            return self.structure
        return self.class_structure.get(class_idx, self.structure)

    def resolve_anchor_effect_size(self) -> float:
        """Convert anchor_effect_size to a numeric effect size."""
        if self.anchor_effect_size is None:
            return 1.0  # default to medium

        if isinstance(self.anchor_effect_size, str):
            effect_map = {
                "small": 0.5,
                "medium": 1.0,
                "large": 1.5,
            }
            return effect_map[self.anchor_effect_size]

        return float(self.anchor_effect_size)

    def summary(self) -> str:
        """Return human-readable summary in medical terms.

        Returns:
            Formatted summary of cluster configuration.
        """
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"Biomarker Cluster: {self.label or 'Unnamed'}")
        lines.append("=" * 60)
        lines.append("")

        # Cluster structure
        lines.append("Cluster Structure:")
        lines.append(
            f"  Number of markers: {self.n_cluster_features} (1 anchor + {self.n_cluster_features - 1} proxies)"
        )
        lines.append(f"  Correlation strength: rho={self.rho} ({self.structure})")

        # Interpret correlation
        if self.rho < 0.3:
            corr_desc = "weak/independent"
        elif self.rho < 0.6:
            corr_desc = "moderate"
        elif self.rho < 0.8:
            corr_desc = "strong"
        else:
            corr_desc = "very strong (pathway-like)"
        lines.append(f"  Interpretation: {corr_desc} biological coupling")
        lines.append("")

        # Anchor properties
        lines.append("Anchor Marker:")
        lines.append(f"  Role: {self.anchor_role}")

        if self.anchor_role == "informative":
            effect_value = self.resolve_anchor_effect_size()
            effect_str = self.anchor_effect_size if self.anchor_effect_size else "medium"
            lines.append(f"  Effect size: {effect_str} (value={effect_value})")

            if self.anchor_class is not None:
                lines.append(f"  Predicts: class {self.anchor_class}")

            # Medical interpretation of effect size
            if effect_value < 0.7:
                lines.append("  → Subtle signal (large sample needed)")
            elif effect_value < 1.2:
                lines.append("  → Moderate signal (typical biomarker)")
            else:
                lines.append("  → Strong signal (easy to detect)")
        else:
            lines.append("  → Random noise (no signal)")

        lines.append("")
        lines.append(f"Random seed: {self.random_state if self.random_state else 'from dataset'}")

        if self.class_rho:
            lines.append(f"Baseline rho for other classes: {self.rho_baseline}")
            lines.append("Per-class overrides:")
            for k, r in sorted(self.class_rho.items()):
                s = self.class_structure.get(k, self.structure) if self.class_structure else self.structure
                lines.append(f"  class {k}: rho={r} ({s})")

        return "\n".join(lines)

    def __str__(self) -> str:
        """Concise representation for quick reference."""
        parts = [f"n_cluster_features={self.n_cluster_features}", f"rho={self.rho}"]
        if self.anchor_role != "informative":
            parts.append(self.anchor_role)
        if self.anchor_effect_size:
            parts.append(f"effect={self.anchor_effect_size}")
        if self.label:
            parts.append(f"'{self.label}'")
        return f"CorrCluster({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------


class DatasetConfig(BaseModel):
    """Configuration for synthetic dataset generation.

    Overview
    --------
    This model defines the *input-level* controls for building a synthetic dataset.
    It combines:
      - Base role counts: `n_informative` and `n_noise`
      - Correlated clusters: `corr_clusters` (each 1 anchor + (k−1) proxies)
      - Class definitions: `class_configs` (with per-class n_samples and labels)
      - Optional batch effects

    Derived quantities
    ------------------
    These attributes are **derived** and must not be passed by the user:

    - ``n_samples``  = sum(c.n_samples for c in class_configs)
    - ``n_classes``  = len(class_configs)
    - ``n_features`` = n_informative + n_noise + proxies_from_clusters

      where

        proxies_from_clusters = sum(max(0, k - 1) for each CorrClusterConfig
                                    with n_cluster_features = k)

    Labels
    ------
    - If a ClassConfig label is None or "", it is auto-filled as "class_{idx}".
    - `class_labels` returns the list of resolved labels.

    Validation & normalization
    --------------------------
    - A `mode="before"` validator:
        * forbids manual `n_samples`, `n_classes`, `n_features`
        * normalizes `class_sep`:
             - scalar → broadcast to length `n_classes - 1`
             - sequence → checked for numeric entries and length `n_classes - 1`
    - A `mode="after"` validator:
        * checks:
             - `n_informative >= #informative_anchors`
             - `n_noise       >= #noise_anchors`
             - `corr_between` in [-1, 1]
             - `anchor_class` indices are < `n_classes`

    Naming policy
    -------------
    - If `feature_naming="prefixed"`:
        * Free informative:  i1, i2, ...
        * Free noise:        n1, n2, ...
        * Correlated:        corr{cid}_anchor, corr{cid}_2, ..., corr{cid}_k
    - If `feature_naming="simple"`, a generic `feature_{i}` scheme is used.


    Examples (counting)
    -------------------
    1) One cluster k=4 with an informative anchor, plus n_informative=3, n_noise=2
       proxies_from_clusters = (4−1) = 3
       n_features_expected   = 3 + 2 + 3 = 8
       Breakdown:
         - informative_anchors = 1  → free_informative = 3 − 1 = 2
         - noise_anchors = 0        → free_noise       = 2 − 0 = 2

    2) Two clusters: k=5 (informative anchor), k=3 ("noise" anchor), base n_informative=4, n_noise=3
       proxies_from_clusters = (5−1) + (3−1) = 6
       n_features_expected   = 4 + 3 + 6 = 13
       Breakdown:
         - informative_anchors = 1  → free_informative = 4 − 1 = 3
         - noise_anchors = 1        → free_noise       = 3 − 1 = 2

    Example usage:
    --------------
        >>> cfg = DatasetConfig(
        ...     n_informative=5,
        ...     n_noise=3,
        ...     class_configs=[
        ...         ClassConfig(n_samples=50, label="healthy"),
        ...         ClassConfig(n_samples=50, label="diseased"),
        ...     ],
        ...     corr_clusters=[
        ...         CorrClusterConfig(
        ...             n_cluster_features=4,
        ...             rho=0.8,
        ...             anchor_role="informative",
        ...             anchor_effect_size="medium",
        ...             anchor_class=1,
        ...             label="Metabolic Pathway A"
        ...         ),
        ...         CorrClusterConfig(
        ...             n_cluster_features=3,
        ...             rho=0.5,
        ...             anchor_role="noise",
        ...             label="Random Noise Cluster"
        ...         )
        ...     ],
        ...     corr_between=0.1,
        ...     noise_distribution="normal",
        ...     noise_distribution_params={"loc": 0, "scale": 1},
        ...     feature_naming="prefixed",
        ...     random_state=42
        ... )

    See Also:
    --------
    CorrClusterConfig  : Correlated cluster settings (size, rho, anchor role/effect/class)
    DatasetMeta        : Output ground-truth meta (indices for anchors, proxies, cluster layout)
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    # Core dataset structure
    n_informative: int = Field(default=2, ge=0)
    n_noise: int = Field(default=0, ge=0)

    # Multi-class controls
    class_configs: list[ClassConfig] = Field(
        [ClassConfig(n_samples=30, label="healthy"), ClassConfig(n_samples=30, label="diseased")], min_length=2
    )
    class_sep: list[float] = Field(
        default_factory=lambda: [1.5],
        description="Class separation values (normalized to length n_classes - 1).",
    )

    # Noise distribution (NumPy Generator API)
    noise_distribution: DistributionType = "normal"
    noise_distribution_params: dict[str, Any] = Field(default_factory=lambda: {"loc": 0, "scale": 1})

    # Naming
    feature_naming: Literal["prefixed", "simple"] = "prefixed"
    prefix_informative: str = "i"
    prefix_noise: str = "n"
    prefix_corr: str = "corr"

    # Correlated structure
    corr_clusters: list[CorrClusterConfig] = Field(default_factory=list)
    corr_between: float = 0.0  # correlation between different clusters/roles (0 = independent)

    # Batch effects
    batch: BatchEffectsConfig | None = None

    # Global seed
    random_state: int | None = None

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _iter_cluster_dicts(
        raw_config: Mapping[str, Any],
    ) -> Iterable[Mapping[str, Any]]:
        """Yield cluster dicts from raw_config, regardless of item type.

        This helper is kept for potential external use (e.g., pre-inspection of
        raw YAML configs). It is not used in the main validation path.
        """
        clusters: Any = raw_config.get("corr_clusters")  # list[dict] / list[CorrClusterConfig] / None
        if not clusters:
            return []
        out: list[Mapping[str, Any]] = []
        for cc in clusters:
            if isinstance(cc, CorrClusterConfig):
                out.append(cc.model_dump())
            elif isinstance(cc, Mapping):
                out.append(cc)
            else:
                raise TypeError(
                    "corr_clusters entries must be Mapping or CorrClusterConfig, " f"got {type(cc).__name__}"
                )
        return out

    @classmethod
    def _validate_sep_value(cls, class_separation: Any) -> float:
        """Validate a single class separation value (numeric & finite)."""
        try:
            fv = float(class_separation)
        except (TypeError, ValueError) as e:
            raise TypeError(f"class_sep entries must be numeric, got {class_separation!r}") from e
        if not np.isfinite(fv):
            raise ValueError(f"class_sep entries must be finite numbers, got {class_separation!r}")
        return fv

    # ------------------------------------------------------ before validator

    @model_validator(mode="before")
    @classmethod
    def _normalize_and_validate(cls, data: Any) -> Any:
        """Normalize incoming data BEFORE model construction.

        - Forbids manual `n_samples`, `n_classes`, `n_features`.
        - Normalizes `class_sep` to a list of length `n_classes - 1`.
        """
        if isinstance(data, cls):
            return data

        if not isinstance(data, Mapping):
            raise TypeError(f"DatasetConfig expects a mapping-like raw_config, got {type(data).__name__}")

        d: dict[str, Any] = dict(data)

        # Forbid manual override of derived attributes
        for forbidden in ("n_samples", "n_classes", "n_features"):
            if forbidden in d:
                raise ValueError(
                    f"{forbidden} is derived from class_configs/corr_clusters and "
                    "must not be set manually on DatasetConfig."
                )

        classes = d.get("class_configs")
        if not isinstance(classes, Sequence) or isinstance(classes, (str, bytes)):
            raise TypeError("class_configs must be a non-string sequence of class definitions.")

        n_classes = len(classes)
        if n_classes < 2:
            raise ValueError(f"At least two classes are required, got {n_classes}.")

        # Normalize class_sep:
        # - scalar → broadcast
        # - sequence → validate entries and length
        raw_sep = d.get("class_sep", [1.5])
        if isinstance(raw_sep, (int, float)):
            sep_list = [cls._validate_sep_value(raw_sep)] * (n_classes - 1)
        elif isinstance(raw_sep, Sequence) and not isinstance(raw_sep, (str, bytes)):
            sep_list = [cls._validate_sep_value(v) for v in raw_sep]
        else:
            raise TypeError(f"class_sep must be a number or sequence, got {type(raw_sep).__name__}")

        if len(sep_list) != n_classes - 1:
            raise ValueError(f"class_sep length must be n_classes - 1 ({n_classes - 1}), " f"got {len(sep_list)}.")

        d["class_sep"] = sep_list
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

    # ------------------------------------------------------ after validators
    @model_validator(mode="after")
    def _enforce_minimum_informative(self):
        """Ensure n_informative >= number of informative anchors."""
        required = self.count_informative_anchors()
        if self.n_informative < required:
            old = self.n_informative
            object.__setattr__(self, "n_informative", required)
            warnings.warn(
                f"[DatasetConfig] n_informative was increased from {old} to {required} "
                f"because your correlated clusters define {required} informative anchors.",
                UserWarning,
            )
        return self

    @model_validator(mode="after")
    def _auto_generate_labels(self):
        """Auto-generate labels as 'class_{idx}' if not provided."""
        for idx, cls_cfg in enumerate(self.class_configs):
            if cls_cfg.label is None or cls_cfg.label == "":
                # ClassConfig is a BaseModel, so we need object.__setattr__
                object.__setattr__(cls_cfg, "label", f"class_{idx}")
        return self

    @model_validator(mode="after")
    def __post_init__(self):
        """Sanity checks tying together anchors, counts, and classes."""
        if self.n_noise < 0:
            raise ValueError("n_noise must be >= 0.")
        if self.n_informative < 0:
            raise ValueError("n_informative must be >= 0.")

        inf_anchors = self.count_informative_anchors()
        noise_anchors = self.count_noise_anchors()
        if self.n_informative < inf_anchors:
            raise ValueError(f"n_informative ({self.n_informative}) < number of informative anchors ({inf_anchors}).")
        if self.n_noise < noise_anchors:
            raise ValueError(f"n_noise ({self.n_noise}) < number of noise anchors ({noise_anchors}).")

        # Corr-between range sanity check
        if not (-1.0 <= float(self.corr_between) <= 1.0):
            raise ValueError(f"corr_between must lie in [-1, 1], got {self.corr_between}.")

        # anchor_class indices must be < n_classes
        max_idx = self.n_classes - 1
        for cluster in self.corr_clusters or []:
            if cluster.anchor_class is not None and cluster.anchor_class > max_idx:
                raise ValueError(
                    f"CorrClusterConfig.anchor_class={cluster.anchor_class} "
                    f"but only {self.n_classes} classes are defined (max index {max_idx})."
                )

        return self

    # ------------------------------------------------------ convenience API

    @classmethod
    def from_yaml(cls, path: str) -> DatasetConfig:
        """Load from YAML and validate via the same pipeline."""
        import yaml  # local import to keep core dependencies lean

        with open(path, encoding="utf-8") as f:
            raw_config: dict[str, Any] = yaml.safe_load(f) or {}
        return cls.model_validate(raw_config)

    # ------------------------------ anchor / proxy accounting ----------------

    def count_informative_anchors(self) -> int:
        """Count clusters whose anchor contributes as 'informative'.

        Note: This is a subset of n_informative, not a separate count.

        Returns:
            The number of clusters with anchor_role == "informative".

        Note:
            If you want the number of *additional* features contributed by clusters,
            use `self._proxies_from_clusters(self.corr_clusters)`.
        """
        return sum(1 for c in (self.corr_clusters or []) if c.anchor_role == "informative")

    def count_noise_anchors(self) -> int:
        """Count clusters whose anchor is 'noise' (non-informative anchor).

        Returns:
            The number of clusters with anchor_role == "noise".
        """
        return sum(1 for c in (self.corr_clusters or []) if c.anchor_role == "noise")

    @staticmethod
    def _proxies_from_clusters(
        clusters: Iterable[CorrClusterConfig] | None,
    ) -> int:
        """Number of additional features contributed by all clusters.

        For a cluster of size k, proxies = max(0, k - 1) regardless of anchor_role.
        """
        if not clusters:
            return 0
        return sum(max(0, int(c.n_cluster_features) - 1) for c in clusters)

    @property
    def n_informative_free(self) -> int:
        """Informative features outside clusters (excludes informative anchors)."""
        return max(self.n_informative - self.count_informative_anchors(), 0)

    @property
    def n_noise_free(self) -> int:
        """Independent noise features (excludes noise anchors)."""
        return max(self.n_noise - self.count_noise_anchors(), 0)

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
        """Total number of features (informative + noise + cluster proxies)."""
        proxies = self._proxies_from_clusters(self.corr_clusters)
        return int(self.n_informative + self.n_noise + proxies)

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

    # ------------------------------ summary / breakdown ----------------------

    def breakdown(self) -> dict[str, int]:
        """Structured feature counts incl. cluster proxies and anchor split.

        Returns:
            A dict with keys:
            - n_informative_total
            - n_informative_anchors
            - n_informative_free
            - n_noise_total
            - n_noise_anchors
            - n_noise_free
            - proxies_from_clusters
            - n_features
        """
        proxies = self._proxies_from_clusters(self.corr_clusters)
        n_inf_anchors = self.count_informative_anchors()
        n_noise_anchors = self.count_noise_anchors()
        return {
            "n_informative_total": int(self.n_informative),
            "n_informative_anchors": int(n_inf_anchors),
            "n_informative_free": int(max(self.n_informative - n_inf_anchors, 0)),
            "n_noise_total": int(self.n_noise),
            "n_noise_anchors": int(n_noise_anchors),
            "n_noise_free": int(max(self.n_noise - n_noise_anchors, 0)),
            "proxies_from_clusters": int(proxies),
            "n_features": int(self.n_features),
        }

    def summary(self, *, per_cluster: bool = False, as_markdown: bool = False) -> str:
        """Return a human-readable summary of the configuration.

        Args:
            per_cluster: Include one line per cluster (size/role/rho/etc.).
            as_markdown: Render as a Markdown table-like text.

        Returns:
            A formatted string summarizing the feature layout and counts.
        """
        b = self.breakdown()
        lines: list[str] = []

        if as_markdown:
            lines.append("### Feature breakdown")
            lines.append("")
            lines.append("| key | value |")
            lines.append("|-----|-------|")
            for k in [
                "n_informative_total",
                "n_informative_anchors",
                "n_informative_free",
                "n_noise_total",
                "n_noise_anchors",
                "n_noise_free",
                "proxies_from_clusters",
                "n_features",
            ]:
                lines.append(f"| {k} | {b[k]} |")
        else:
            lines.append("Feature breakdown")
            lines.append(f"- n_informative_total    : {b['n_informative_total']}")
            lines.append(f"- n_informative_anchors  : {b['n_informative_anchors']}")
            lines.append(f"- n_informative_free     : {b['n_informative_free']}")
            lines.append(f"- n_noise_total          : {b['n_noise_total']}")
            lines.append(f"- n_noise_anchors        : {b['n_noise_anchors']}")
            lines.append(f"- n_noise_free           : {b['n_noise_free']}")
            lines.append(f"- proxies_from_clusters  : {b['proxies_from_clusters']}")
            lines.append(f"- n_features             : {b['n_features']}")

        if per_cluster and self.corr_clusters:
            if as_markdown:
                lines.append("")
                lines.append("### Clusters")
                lines.append("")
                lines.append("| id | size | role | rho | structure | label | proxies |")
                lines.append("|----|------|------|-----|-----------|-------|---------|")
            else:
                lines.append("Clusters:")

            for i, c in enumerate(self.corr_clusters, start=1):
                proxies = max(0, c.n_cluster_features - 1)
                label = c.label or ""
                if as_markdown:
                    lines.append(
                        f"| {i} | {c.n_cluster_features} | {c.anchor_role} | "
                        f"{c.rho} | {c.structure} | {label} | {proxies} |"
                    )
                else:
                    lines.append(
                        f"- #{i}: n_cluster_features={c.n_cluster_features}, "
                        f"role={c.anchor_role}, rho={c.rho}, structure={c.structure}, "
                        f"label={label}, proxies={proxies}"
                    )

        return "\n".join(lines)
