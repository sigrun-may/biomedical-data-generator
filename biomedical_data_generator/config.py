# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Configuration models for the dataset generator."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from enum import Enum
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RawConfig: TypeAlias = Mapping[str, Any]
MutableRawConfig: TypeAlias = MutableMapping[str, Any]
AnchorMode: TypeAlias = Literal["equalized", "strong"]


class NoiseDistribution(str, Enum):
    """Supported noise distributions for generating noise features.

    Semantics:
    - `noise_distribution`: choice of distribution used *during sampling*:
        * normal  → Gaussian (normal) distribution
        * laplace → Laplace distribution (heavy tails; useful for simulating outliers)
        * uniform → Uniform distribution (bounded noise)
    - `noise_scale`: scale parameter used *during sampling*:
        * normal  → standard deviation σ
        * laplace → diversity b
        * uniform → half-width (samples are in [−noise_scale, +noise_scale] by default)
      Distribution-specific values in `noise_params` (e.g. 'scale', 'low', 'high', 'loc')
      take precedence over `noise_scale`.

    - `noise_params` (optional): fine-grained overrides passed to the sampler:
        * normal  → {'loc': float, 'scale': float}
        * uniform → {'low': float, 'high': float}
        * laplace → {'loc': float, 'scale': float}
      If a key is omitted, sensible defaults are used:
        * normal  → loc=0.0, scale=noise_scale
        * uniform → low=−noise_scale, high=+noise_scale
        * laplace → loc=0.0, scale=noise_scale

    Args:
        normal: Gaussian (normal) distribution.
        uniform: Uniform distribution.
        laplace: Laplace distribution (heavy tails; useful for simulating outliers).

    Examples:
    --------
        ```python
        from biomedical_data_generator import DatasetConfig, NoiseDistribution
        cfg = DatasetConfig(
            n_samples=100,
            n_features=10,
            n_informative=2,
            n_noise=5,
            noise_distribution=NoiseDistribution.laplace,
            noise_params={'loc': 0.0, 'scale': 0.5},
            noise_scale=2.0,
            random_state=42
        )
        ```
    See also:
         - numpy.random.Generator documentation for parameters:
              https://numpy.org/doc/stable/reference/random/generator.html#distributions

    Note:
        - The chosen distribution `noise_distribution` is applied to all noise features uniformly.
        - The `noise_params` can be used to fine-tune the noise characteristics.
        - For teaching outliers/heavy tails, `laplace` is useful; for bounded noise, use `uniform`.

    """

    normal = "normal"
    uniform = "uniform"
    laplace = "laplace"  # heavy tails; nice to show outliers


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
        n_batches (int):
            Number of batches. Values 0 or 1 effectively disable batch effects.
        effect_strength (float):
            Scale of batch effects.
            - For ``effect_type="additive"``: standard deviation of batch intercepts.
            - For ``effect_type="multiplicative"``: standard deviation of the
              multiplicative factor (applied as ``X' = X * (1 + b_batch)``).
        effect_type (Literal["additive", "multiplicative"]):
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
            - ``"informative"``: apply only to informative features
              (indices passed from the generator).
            - list of ints: explicit 0-based column indices.
        proportions (list[float] | None):
            Optional target proportions for batch sizes. Values are normalized
            to sum to 1. If ``None``, batches are (approximately) equal in size.
        random_state (int | None):
            Optional seed **specific to batch effects**.
            - ``None`` (recommended): reuse the dataset-level RNG created from
              ``DatasetConfig.random_state``.
            - ``int``: use a dedicated RNG ``np.random.default_rng(random_state)``
              for batch assignment and effects. This allows advanced users to
              keep batch effects reproducible even if the dataset-level seed
              changes.

    Examples:
    --------
    >>> # Simple random batches, all controlled by DatasetConfig.random_state
    >>> BatchEffectsConfig(n_batches=3, effect_strength=0.5)

    >>> # Confounded batches (recruitment bias)
    >>> BatchEffectsConfig(
    ...     n_batches=2,
    ...     effect_strength=0.8,
    ...     confounding_with_class=0.9,
    ... )

    >>> # Unbalanced batches with an explicit batch seed
    >>> BatchEffectsConfig(
    ...     n_batches=3,
    ...     proportions=[0.5, 0.3, 0.2],
    ...     effect_strength=0.5,
    ...     random_state=123,
    ... )
    """

    n_batches: int = Field(default=0, ge=0)  # 0 or 1 => no batch effect
    effect_strength: float = Field(default=0.5, gt=0)  # std of batch effects
    effect_type: Literal["additive", "multiplicative"] = "additive"
    confounding_with_class: float = Field(default=0.0, ge=0.0, le=1.0)  # in [0,1]
    affected_features: list[int] | Literal["all", "informative"] = "all"  # 0-based column indices; "all" => all
    proportions: list[float] | None = None  # optional proportions for batches
    random_state: int | None = None

    @field_validator("proportions")
    @classmethod
    def validate_proportions(cls, v):
        """Ensure proportions sum to ~1."""
        if v is not None:
            if not np.isclose(sum(v), 1.0, atol=1e-6):
                raise ValueError(f"proportions must sum to 1, got {sum(v)}")
        return v


# class BatchEffectsConfig(BaseModel):
#     """Configuration for simulating batch effects.
#
#     Simulate batch effects by adding random intercepts to specified columns. The intercepts are drawn from a normal
#     distribution with mean 0 and standard deviation `effect_strength`. Optionally, the batch assignment can
#     be confounded with the class labels.
#
#     Args:
#         n_batches (int): Number of batches (0 or 1 means no batch effect).
#         effect_strength (float): Standard deviation of the batch intercepts.
#         effect_type (Literal["additive", "multiplicative"]): Type of batch effect.
#             - "additive": Additive intercepts (default).
#             - "multiplicative": Multiplicative scaling factors.
#         confounding_with_class: Degree of confounding between batch and class (0.0-1.0).
#             Controls how strongly batch assignment correlates with class labels,
#             simulating **recruitment bias** in multi-center studies.
#
#             The parameter determines the probability that samples from the same
#             class are assigned to the same batch:
#
#             - 0.0 = Independent: Batch assignment ignores class labels
#               → Each batch has approximately equal class proportions
#
#             - 0.5 = Moderate correlation: Samples from the same class are
#               moderately likely to end up in the same batch
#               → Batches show noticeable class imbalance
#
#             - 1.0 = Perfect correlation: Samples from the same class are
#               always assigned to the same batch
#               → Each batch contains only one class (if n_batches ≥ n_classes)
#
#             This controls **sampling bias** (who gets recruited where), while
#             `effect_strength` controls **technical variation** (measurement differences).
#         affected_features (list[int] | Literal["all", "informative"]): Features to affect.
#             "all" affects all features, "informative" affects only informative features,
#             or provide list of 0-based column indices.
#         proportions (list[float] | None): Optional proportions for batch sizes. Must sum to 1.
#         random_state (int | None): Random seed for reproducibility.
#
#     Examples:
#         >>> # Simple random batches
#         >>> BatchEffectsConfig(n_batches=3, effect_strength=0.5)
#
#         >>> # Confounded batches (recruitment bias)
#         >>> BatchEffectsConfig(
#         ...     n_batches=2,
#         ...     effect_strength=0.8,
#         ...     confounding_with_class=0.9
#         ... )
#
#         >>> # Unbalanced batches
#         >>> BatchEffectsConfig(
#         ...     n_batches=3,
#         ...     proportions=[0.5, 0.3, 0.2],
#         ...     effect_strength=0.5
#         ... )
#     """
#
#     n_batches: int = Field(default=0, ge=0)  # 0 or 1 => no batch effect
#     effect_strength: float = Field(default=0.5, gt=0)  # std of batch intercepts
#     effect_type: Literal["additive", "multiplicative"] = "additive"
#     confounding_with_class: float = Field(default=0.0, ge=0.0, le=1.0)  # in [0,1]
#     affected_features: list[int] | Literal["all", "informative"] = "all"  # 0-based column indices; "all" => all
#     proportions: list[float] | None = None  # optional proportions for batches
#     random_state: int | None = None
#
#     @field_validator("proportions")
#     @classmethod
#     def validate_proportions(cls, v):
#         """Ensure proportions sum to ~1."""
#         if v is not None:
#             if not np.isclose(sum(v), 1.0, atol=1e-6):
#                 raise ValueError(f"proportions must sum to 1, got {sum(v)}")
#         return v
#


class CorrClusterConfig(BaseModel):
    """Correlated feature cluster simulating coordinated biomarker patterns.

    A cluster represents a group of biomarkers that move together, such as
    markers in a metabolic pathway or proteins in a signaling cascade. One marker
    acts as the "anchor" (driver), while the others are "proxies" (followers).

    Args:
        n_cluster_features: Number of biomarkers in the cluster (including anchor). Must be >= 1.
        rho: Correlation strength between biomarkers in the cluster.
            - 0.0 = independent
            - 0.5 = moderate correlation
            - 0.8+ = strong correlation (typical for pathway markers)
            - Range: [0, 1) for equicorrelated; (-1, 1) for toeplitz
            Default is 0.8.
        structure: Pattern of correlation within the cluster.
            - "equicorrelated": all pairs have the same correlation (default)
            - "toeplitz": correlation decreases with distance
        class_structure: Mapping of class index to correlation structure.
        class_rho: Mapping of class index to correlation strength.
        rho_baseline: Baseline correlation for other classes if class_rho is set. Default is 0.0 (independent).
        anchor_role: Biological relevance of the anchor marker.
            - "informative": true biomarker (predictive of disease)
            - "noise": random measurement (no biological signal)
        anchor_effect_size: Strength of the anchor's disease association.
            Can be specified as:
            - Preset: "small" (0.5), "medium" (1.0), "large" (1.5)
            - Custom float: any positive value
            - None: defaults to "medium" (1.0)
            Only relevant when `anchor_role="informative"`.
        anchor_class: Disease class that this anchor predicts (0, 1, 2, ...).
            If None, the anchor contributes to all classes. Only used when
            `anchor_role="informative"`.
        random_state: Random seed for reproducibility of this specific cluster.
            If None, uses the global dataset seed.
        label: Descriptive name for documentation (e.g., "Inflammation markers").

    Examples:
    --------
        Strong inflammatory pathway in diseased patients:

        >>> inflammation = CorrClusterConfig(
        ...     n_cluster_features=5,
        ...     rho=0.8,
        ...     anchor_role="informative",
        ...     anchor_effect_size="large",
        ...     anchor_class=1,  # disease class
        ...     label="Inflammation pathway"
        ... )

        Weak disease signal with custom effect size:

        >>> weak_signal = CorrClusterConfig(
        ...     n_cluster_features=4,
        ...     rho=0.5,
        ...     anchor_role="informative",
        ...     anchor_effect_size=0.3,  # custom weak effect
        ...     label="Subtle biomarkers"
        ... )

    Notes:
    -----
        Medical interpretation:
        - **Anchor**: The primary biomarker (e.g., CRP in inflammation)
        - **Proxies**: Secondary markers that follow the anchor (e.g., IL-6, TNF-α)
        - **rho=0.8**: Typical for tightly regulated biological pathways
        - **rho=0.5**: Moderate biological coupling
        - **effect_size="large"**: Strong disease association (easy to detect)
        - **effect_size="small"**: Subtle signal (requires large sample size)

        Technical details:
        - Cluster contributes `n_cluster_features` features to the dataset
        - Anchor appears first, followed by `(n_cluster_features-1)` proxies
        - Only the anchor has predictive power; proxies are correlated distractors
        - Proxies count as additional features beyond n_informative/n_noise

    See Also:
    --------
        DatasetConfig : Overall dataset configuration
        generate_dataset : Main generation function
    """

    model_config = ConfigDict(extra="forbid")

    # Core cluster structure and correlation settings
    n_cluster_features: int = Field(..., ge=1, description="Number of biomarkers in cluster")

    # global correlation (default mode)
    structure: Literal["equicorrelated", "toeplitz"] = Field(
        default="equicorrelated", description="Default correlation structure for all classes"
    )
    rho: float = Field(default=0.8, description="Default correlation strength for all classes")

    # per-class correlation (optional)
    class_structure: dict[int, Literal["equicorrelated", "toeplitz"]] | None = Field(
        default=None, description="Per-class correlation structure (overrides 'structure' for specified classes)"
    )
    class_rho: dict[int, float] | None = Field(
        default=None, description="Per-class correlation strength (activates class-specific mode)"
    )
    rho_baseline: float = Field(
        default=0.0, ge=-1.0, lt=1.0, description="Fallback correlation for classes not in class_rho"
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
            raise ValueError(f"n_cluster_features must be >= 1, got {v}")
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
                raise ValueError(f"class_rho keys must be >= 0, got {cls_idx}")
            if not (lower < float(rho_val) < 1.0):
                raise ValueError(
                    f"class_rho[{cls_idx}]={rho_val} invalid for {structure} (p={p}); "
                    f"require {lower:.6f} < rho < 1."
                )
        return v

    @field_validator("class_structure")
    @classmethod
    def _validate_class_structure(cls, v: dict[int, str] | None) -> dict[int, str] | None:
        """Validate per-class structure keys."""
        if v is None:
            return v

        for cls_idx in v.keys():
            if cls_idx < 0:
                raise ValueError(f"class_structure keys must be >= 0, got {cls_idx}")
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

    def is_class_specific(self) -> bool:
        """Check if this cluster uses class-specific correlation.

        Returns:
            True if class_rho is set (activates class-specific mode).
        """
        return self.class_rho is not None

    def get_rho_for_class(self, class_idx: int) -> float:
        """Get correlation strength for a specific class.

        Args:
            class_idx: Class label (0, 1, 2, ...).

        Returns:
            Correlation strength for this class.
        """
        if not self.is_class_specific():
            return self.rho

        assert self.class_rho is not None  # type guard
        return self.class_rho.get(class_idx, self.rho_baseline)

    def get_structure_for_class(self, class_idx: int) -> Literal["equicorrelated", "toeplitz"]:
        """Get correlation structure for a specific class.

        Args:
            class_idx: Class label (0, 1, 2, ...).

        Returns:
            Structure type for this class.
        """
        if self.class_structure is None:
            return self.structure

        return self.class_structure.get(class_idx, self.structure)

    @field_validator("anchor_effect_size")
    @classmethod
    def _validate_effect_size(cls, v) -> Literal["small", "medium", "large"] | float | None:
        """Validate effect size is either a preset or positive float."""
        if v is None:
            return v

        # Preset string
        if isinstance(v, str):
            if v not in ("small", "medium", "large"):
                raise ValueError(f"anchor_effect_size must be 'small', 'medium', or 'large', got '{v}'")
            return cast(Literal["small", "medium", "large"], v)

        # Custom float
        try:
            val = float(v)
            if val <= 0:
                raise ValueError(f"anchor_effect_size must be > 0, got {val}")
            return val
        except (TypeError, ValueError) as e:
            raise ValueError(f"anchor_effect_size must be 'small'/'medium'/'large' or positive float, got {v}") from e

    @field_validator("anchor_class")
    @classmethod
    def _validate_anchor_class(cls, v: int | None) -> int | None:
        """Ensure disease class is non-negative if specified."""
        if v is not None and v < 0:
            raise ValueError(f"anchor_class must be >= 0 or None, got {v}")
        return v

    def resolve_anchor_effect_size(self) -> float:
        """Convert anchor_effect_size to numeric value.

        Returns:
            Numeric effect size for calculations.
            - "small" → 0.5
            - "medium" → 1.0  (default)
            - "large" → 1.5
            - custom float → as specified

        Examples:
        --------
            >>> c = CorrClusterConfig(n_cluster_features=3, rho=0.7, anchor_effect_size="large")
            >>> c.resolve_anchor_effect_size()
            1.5

            >>> c = CorrClusterConfig(n_cluster_features=3, rho=0.7, anchor_effect_size=0.8)
            >>> c.resolve_anchor_effect_size()
            0.8

            >>> c = CorrClusterConfig(n_cluster_features=3, rho=0.7)  # default
            >>> c.resolve_anchor_effect_size()
            1.0
        """
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
        lines.append(f"{'=' * 60}")
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


class DatasetConfig(BaseModel):
    """Configuration for synthetic dataset generation.

    Overview
    --------
    This model defines the *input-level* controls for building a synthetic dataset.
    It combines:
      - Base role counts: `n_informative` and `n_noise`
      - Correlated clusters: `corr_clusters` (each with 1 anchor + (k−1) non-anchors)
      - Class distribution, separation strength, naming, and optional batch effects

        Terminology
    -----------
    - **Anchor**: the designated first feature of a cluster; can be informative or noise.
    - **Proxy**: any correlated, non-anchor member of a cluster.
    - **Free informative/noise**: features outside clusters, after accounting for anchors.

    Anchors vs. Proxies — Counting Semantics
    ----------------------------------------
    • Each correlated cluster contributes exactly 1 **anchor** (the first feature in the cluster)
      and (k−1) **non-anchor** members (k = n_cluster_features).

    • The anchor has a role `anchor_role` ∈ {"informative", "noise"}:
        - If `anchor_role == "informative"`, the anchor **counts toward `n_informative`**.
        - If `anchor_role == "noise"`, the anchor **counts toward `n_noise`**.

    • **Proxies** are the non-anchor members of a cluster. They are *derived* from
      `corr_clusters` and are **not configured directly** (there is no input field like `n_proxy`).
      They are **added on top** of `n_informative + n_noise` with the formula:
          proxies_from_clusters = Σ_over_clusters (n_cluster_features − 1)

    Expected feature count (strict)
    -------------------------------
    The minimal required total number of features is:
        n_features_expected = n_informative + n_noise + proxies_from_clusters
    with:
        proxies_from_clusters = sum(max(0, c.n_cluster_features - 1) for c in corr_clusters)

    The strict `@model_validator(mode="before")` computes this minimum and:
      - fills `n_features` if omitted,
      - enforces `n_features >= n_features_expected`,
      - raises a clear error message if too small.
    (See `_required_n_features` and the “before” normalizer for details.)

    Derived “free” counts (interpretation aid)
    ------------------------------------------
    For reasoning and naming it is often helpful to split base counts into:
      - `n_informative_anchors` = number of clusters with `anchor_role="informative"`
      - `n_informative_free`   = `n_informative - n_informative_anchors`
      - `n_noise_anchors`      = number of clusters with `anchor_role="noise"`
      - `n_noise_free`         = `n_noise - n_noise_anchors`
    These “free” features are outside clusters; the cluster anchors are part of the base counts.

    Validation & Normalization
    --------------------------
    - **Before-construction validator** (`mode="before"`):
        * Computes `n_features_expected` from raw input
        * Fills or checks `n_features` accordingly (strict lower bound)
        * Performs basic shape/type checks on class-related fields
    - **After-construction validator** (`mode="after"`):
        * Sanity checks for noise distribution and parameters
        * (Recommended) Invariants:
            - `n_informative >= #informative_anchors`
            - `n_noise >= #noise_anchors`

    Naming policy
    -------------
    - If `feature_naming="prefixed"`:
        * Free informative features:  i1, i2, ...
        * Free noise features:        n1, n2, ...
        * Correlated clusters:        corr{cid}_anchor, corr{cid}_2, ..., corr{cid}_k
    - If `feature_naming="simple"`, a generic `feature_{i}` scheme is used.

    Relationship to DatasetMeta (output, ground truth)
    --------------------------------------------------
    The generation step produces a `DatasetMeta` object describing resolved structure:
      - `informative_idx`: indices of all informative features (anchors + free informative)
      - `noise_idx`:      indices of all independent noise features (free noise)
      - `corr_cluster_indices`: mapping cluster_id → list of column indices
      - `anchor_idx`: mapping cluster_id → anchor column (or None for degenerate forms)
    This meta is the single source of truth for downstream visualization/evaluation.

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

    See Also:
    --------
    CorrClusterConfig  : Correlated cluster settings (size, rho, anchor role/effect/class)
    DatasetMeta        : Output ground-truth meta (indices for anchors, proxies, cluster layout)
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    n_samples: int = 100
    n_features: int | None = None
    n_informative: int = 2
    n_noise: int = 0
    noise_distribution: NoiseDistribution = NoiseDistribution.normal
    noise_scale: float = 1.0
    noise_params: Mapping[str, Any] | None = Field(default=None)

    # multi-class controls
    n_classes: int = 2
    weights: list[float] | None = None  # will be normalized by generator; only length is checked here
    class_counts: dict[int, int] | None = None  # exact class sizes; overrides weights if provided
    class_sep: float = 1.5

    # naming
    feature_naming: Literal["prefixed", "simple"] = "prefixed"
    prefix_informative: str = "i"
    prefix_proxy: str = "p"
    prefix_noise: str = "n"
    prefix_corr: str = "corr"

    # structure
    corr_clusters: list[CorrClusterConfig] = Field(default_factory=list)
    corr_between: float = 0.0  # correlation between different clusters/roles (0 = independent)
    anchor_mode: AnchorMode = "equalized"
    effect_size: Literal["small", "medium", "large"] = "medium"  # controls default anchor_effect_size
    batch: BatchEffectsConfig | None = None
    random_state: int | None = None

    # ---------- helpers (typed) ----------
    @staticmethod
    def _iter_cluster_dicts(raw_config: RawConfig) -> Iterable[Mapping[str, Any]]:
        """Yield cluster dicts from raw_config, regardless of whether items are dicts or CorrCluster instances.

        Args:
            raw_config: The raw input config mapping.

        Yields:
        ------
            An iterable of cluster dicts.

        Raises:
        ------
            TypeError: If any entry is neither a dict nor a CorrCluster instance.

        Note:
        This is a static method because it operates on raw input data before model instantiation.
        """
        clusters: Any = raw_config.get("corr_clusters")  # could be None / list[dict] / list[CorrCluster]
        if not clusters:
            return []
        out: list[Mapping[str, Any]] = []
        for cc in clusters:
            if isinstance(cc, CorrClusterConfig):
                out.append(cc.model_dump())
            elif isinstance(cc, Mapping):
                out.append(cc)
            else:
                raise TypeError(f"corr_clusters entries must be Mapping or CorrCluster, got {type(cc).__name__}")
        return out

    @classmethod
    def _required_n_features(cls, raw_config: RawConfig) -> int:
        """Compute the minimal number of features needed based on roles and correlated clusters.

        Assumption
        ----------
        Each cluster of size 'k' contributes (k - 1) *additional* features,
        because its anchor is already counted in the base role counts.
        """
        base = int(raw_config.get("n_informative", 0)) + int(raw_config.get("n_noise", 0))
        extra_from_clusters = 0
        for c in cls._iter_cluster_dicts(raw_config):
            n_cluster_features = int(c.get("n_cluster_features", 0))
            extra_from_clusters += max(0, n_cluster_features - 1)
        return base + extra_from_clusters

    # ---------- validation (strict, no warnings) ----------
    @model_validator(mode="after")
    def __post_init__(self):
        if self.n_noise < 0:
            raise ValueError("n_noise must be >= 0")
        if self.noise_scale <= 0:
            raise ValueError("noise_scale must be > 0")
        # validate uniform bounds if both provided
        if self.noise_distribution == NoiseDistribution.uniform and self.noise_params:
            low = self.noise_params.get("low", None)
            high = self.noise_params.get("high", None)
            if low is not None and high is not None and not (float(low) < float(high)):
                raise ValueError("For uniform noise, require low < high.")

        inf_anchors = self.count_informative_anchors()
        noise_anchors = self.count_noise_anchors()
        if self.n_informative < inf_anchors:
            raise ValueError(f"n_informative ({self.n_informative}) < number of informative anchors ({inf_anchors}).")
        if self.n_noise < noise_anchors:
            raise ValueError(f"n_noise ({self.n_noise}) < number of noise anchors ({noise_anchors}).")
        return self

    @model_validator(mode="before")
    @classmethod
    def _normalize_and_validate(cls, data: Any) -> Any:
        """Normalize incoming raw_config BEFORE model construction.

        This is a 'before' validator, so it works on raw input data. It fills in
        missing n_features and enforces n_features >= required minimum.

        Args:
            cls: The DatasetConfig class.
            data: The raw input data (any mapping-like).

        Returns:
        -------
            A mapping with normalized/validated fields, suitable for model construction.

        Raises:
        ------
            TypeError: If data is not a mapping or if fields have wrong types.
            ValueError: If n_features is too small or if other value constraints are violated.

        Note:
            This does NOT modify the original data dict.
        """
        if isinstance(data, cls):
            # already a DatasetConfig instance
            return data

        # ensure we work on a mutable mapping copy
        if isinstance(data, Mapping):
            d: dict[str, Any] = dict(data)
        else:
            raise TypeError(f"DatasetConfig expects a mapping-like raw_config, got {type(data).__name__}")

        # n_features fill/check
        required = cls._required_n_features(d)
        n_features = d.get("n_features")
        if n_features is None:
            d["n_features"] = required
        else:
            try:
                n_features_int = int(n_features)
            except Exception as e:  # noqa: BLE001
                raise TypeError(f"n_features must be an integer, got {n_features!r}") from e
            if n_features_int < required:
                raise ValueError(
                    f"n_features={n_features_int} is too small; requires at least {required} "
                    f"(given roles + correlated clusters)."
                )
            d["n_features"] = n_features_int

        # n_classes: defer strict check to runtime; keep type sanity here
        try:
            n_classes = int(d.get("n_classes", 2))
        except Exception as e:  # noqa: BLE001
            raise TypeError("n_classes must be an integer") from e
        d["n_classes"] = n_classes

        # weights length (if provided) must match n_classes
        weights = d.get("weights")
        if weights is not None:
            if not isinstance(weights, list) or not all(isinstance(x, (int, float)) for x in weights):
                raise TypeError("weights must be a list of numbers")
            if len(weights) != n_classes:
                raise ValueError(f"weights length ({len(weights)}) must equal n_classes ({n_classes})")

        # class_sep must be finite (basic sanity)
        class_separation = float(d.get("class_sep", 1.0))
        if (
            not (class_separation == class_separation)
            or class_separation == float("inf")
            or class_separation == float("-inf")
        ):
            raise ValueError("class_sep must be a finite float")
        d["class_sep"] = class_separation

        return d

    @field_validator("weights")
    @classmethod
    def _validate_weights(cls, v, info):
        """Non-negative, correct length; normalization happens in the generator."""
        if v is None:
            return v
        n_classes = info.data.get("n_classes", None)
        if n_classes is not None and len(v) != n_classes:
            raise ValueError(f"weights length must equal n_classes (got {len(v)} vs {n_classes})")
        if any(w < 0 for w in v):
            raise ValueError("weights must be non-negative.")
        if all(w == 0 for w in v):
            raise ValueError("at least one weight must be > 0.")
        return v

    @field_validator("class_counts")
    @classmethod
    def _validate_class_counts(cls, v, info):
        """Exact counts must cover all classes and sum to n_samples."""
        if v is None:
            return v
        n_samples = info.data.get("n_samples")
        n_classes = info.data.get("n_classes")
        keys = set(v.keys())
        expected = set(range(n_classes))
        if keys != expected:
            raise ValueError(f"class_counts keys must be {expected} (got {keys}).")
        total = sum(int(c) for c in v.values())
        if total != n_samples:
            raise ValueError(f"sum(class_counts) must equal n_samples (got {total} vs {n_samples}).")
        if any(int(c) < 0 for c in v.values()):
            raise ValueError("class_counts must be non-negative.")
        return {int(k): int(c) for k, c in v.items()}

    # ---------- convenience factories ----------

    @classmethod
    def relaxed(cls, **kwargs: Any) -> DatasetConfig:
        """Create a configuration, silently autofixing n_features to the required minimum.

        Convenience factory that silently 'autofixes' n_features to the required minimum. Prefer this in teaching
        notebooks to avoid interruptions.

        Args:
            **kwargs: Any valid DatasetConfig field.

        Returns:
        -------
            A validated DatasetConfig instance with n_features >= required minimum.

        Note:
            This does NOT modify the original kwargs dict.
        """
        raw = kwargs.get("corr_clusters") or []
        norm_clusters = []
        for c in raw:
            if isinstance(c, CorrClusterConfig):
                norm_clusters.append(c)
            else:
                norm_clusters.append(CorrClusterConfig.model_validate(c))
        kwargs["corr_clusters"] = norm_clusters

        # Compute required n_features
        # n_features = free informative + free noise + sum(n_cluster_features-1 per cluster)
        n_inf = int(kwargs.get("n_informative", 0))
        n_noise = int(kwargs.get("n_noise", 0))
        proxies = sum(cc.n_cluster_features - 1 for cc in norm_clusters)
        required = n_inf + n_noise + proxies

        n_feat = kwargs.get("n_features")
        if n_feat is None or int(n_feat) < required:
            kwargs["n_features"] = required

        return cls(**kwargs)

        # d: dict[str, Any] = dict(kwargs)
        # required = cls._required_n_features(d)
        # nf = d.get("n_features")
        # if nf is None or int(nf) < required:
        #     d["n_features"] = required
        # return cls.model_validate(d)

    @classmethod
    def from_yaml(cls, path: str) -> DatasetConfig:
        """Load from YAML and validate via the same 'before' pipeline.

        Args:
            cls: The DatasetConfig class.
            path: Path to a YAML file.

        Returns:
        -------
            A validated DatasetConfig instance.

        Raises:
        ------
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the file cannot be parsed as YAML.
            pydantic.ValidationError: If the loaded config is invalid.

        Note:
            This requires PyYAML to be installed.
        """
        import yaml  # local import to keep core dependencies lean

        with open(path, encoding="utf-8") as f:
            raw_config: dict[str, Any] = yaml.safe_load(f) or {}
        return cls.model_validate(raw_config)

    # --- Convenience helpers for introspection ---------------------------------

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
    def _proxies_from_clusters(clusters: Iterable[CorrClusterConfig] | None) -> int:
        """Compute the number of additional features contributed by clusters beyond their anchor.

        Number of *additional* features contributed by clusters beyond their anchor. For a cluster of size k,
        proxies = max(0, k - 1) regardless of anchor_role.

        Args:
            clusters: An iterable of CorrCluster instances (or None).

        Returns:
        -------
            The total number of additional features contributed by all clusters.

        Note:
            This is consistent with the required_n_features calculation.
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
            - n_features_expected
            - n_features_configured
        Raises:
            ValueError: If self.n_features is inconsistent (should not happen if validated).
            This is a safeguard against manual tampering with the instance attributes. This should not happen if the
            instance was created via the normal validators. If you encounter this, please report a bug.
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
            "n_features_expected": int(self.n_informative + self.n_noise + proxies),
            "n_features_configured": int(self.n_features or 0),
        }

    def summary(self, *, per_cluster: bool = False, as_markdown: bool = False) -> str:
        """Return a human-readable summary of the configuration.

        Args:
            per_cluster: Include one line per cluster (size/role/rho/etc.).
            as_markdown: Render as a Markdown table-like text.

        Returns:
        -------
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
                "n_noise",
                "proxies_from_clusters",
                "n_features_expected",
                "n_features_configured",
            ]:
                lines.append(f"| {k} | {b[k]} |")
        else:
            lines.append("Feature breakdown")
            lines.append(f"- n_informative_total    : {b['n_informative_total']}")
            lines.append(f"- n_informative_anchors  : {b['n_informative_anchors']}")
            lines.append(f"- n_informative_free     : {b['n_informative_free']}")
            lines.append(f"- n_noise                : {b['n_noise']}")
            lines.append(f"- proxies_from_clusters  : {b['proxies_from_clusters']}")
            lines.append(f"- n_features_expected    : {b['n_features_expected']}")
            lines.append(f"- n_features_configured  : {b['n_features_configured']}")

        if per_cluster and self.corr_clusters:
            header = "| id | size | role | rho | structure | label | proxies |" if as_markdown else "Clusters:"
            if as_markdown:
                lines.append("")
                lines.append("### Clusters")
                lines.append("")
                lines.append(header)
                lines.append("|----|------|------|-----|-----------|-------|---------|")
            else:
                lines.append(header)

            for i, c in enumerate(self.corr_clusters, start=1):
                proxies = max(0, c.n_cluster_features - 1)  # consistent with required_n_features
                label = c.label or ""
                if as_markdown:
                    lines.append(
                        f"| {i} | {c.n_cluster_features} | {c.anchor_role} | {c.rho} | {c.structure} | "
                        f"{label} | {proxies} |"
                    )
                else:
                    lines.append(
                        f"- #{i}: n_cluster_features={c.n_cluster_features}, role={c.anchor_role}, rho={c.rho}, "
                        f"structure={c.structure}, label={label}, proxies={proxies}"
                    )

        return "\n".join(lines)
