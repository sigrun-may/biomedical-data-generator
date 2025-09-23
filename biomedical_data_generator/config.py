# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Configuration models for the dataset generator."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class CorrCluster(BaseModel):
    """Configuration for a correlated feature cluster.

    Attributes
    ----------
        size: Number of features in the cluster.
        rho: Target correlation within the cluster (0 ≤ rho < 1).
        structure: Correlation structure ("equicorrelated" or "ar1").
        anchor_role: Whether the cluster has an anchor that affects y
            ("informative") or is purely latent ("latent").
        anchor_beta: Effect size for the anchor (only if informative).
        anchor_class: Class index (0..n_classes-1) for which the anchor
            contributes positively. Ignored if anchor_role="latent".
        random_state: Optional seed for reproducibility of this cluster.
        label: Optional didactic label for clarity in teaching contexts.
    """

    size: int
    rho: float = 0.7
    structure: Literal["equicorrelated", "ar1"] = "equicorrelated"
    anchor_role: Literal["informative", "latent"] = "latent"
    anchor_beta: float = 1.0
    anchor_class: int | None = 0
    random_state: int | None = None
    label: str | None = None


class DatasetConfig(BaseModel):
    """Configuration for generating a synthetic classification dataset.

    Attributes
    ----------
        n_samples: Number of samples (rows).
        n_features: Total number of features.
        n_informative: Number of informative features (including anchors).
        n_pseudo: Number of pseudo features (correlated proxies or free).
        n_noise: Number of pure noise features.
        class_sep: Global scaling factor for signal strength.
        n_classes: Number of classes (>=2).
        weights: Optional class proportions of length n_classes.
        corr_between: Global coupling between all cluster features.
        corr_clusters: Optional list of correlated feature clusters.
        feature_naming: How to name features ("prefixed" or "generic").
        prefix_informative: Prefix for informative features.
        prefix_corr: Prefix for correlated proxy features.
        prefix_pseudo: Prefix for free pseudo features.
        prefix_noise: Prefix for noise features.
        random_state: Optional global random seed.
    """

    n_samples: int = Field(200, ge=1)
    n_features: int
    n_informative: int
    n_pseudo: int = 0
    n_noise: int = 0
    class_sep: float = 1.0

    # Multiclass extension
    n_classes: int = 2
    weights: Sequence[float] | None = None  # length n_classes or None

    # Correlation
    corr_between: float = 0.0
    corr_clusters: list[CorrCluster] | None = None

    # Naming
    feature_naming: Literal["prefixed", "generic"] = "prefixed"
    prefix_informative: str = "i"
    prefix_corr: str = "corr"
    prefix_pseudo: str = "p"
    prefix_noise: str = "n"

    # Randomness
    random_state: int | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> DatasetConfig:
        """Load a DatasetConfig from a YAML file."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)  # explicit v2 API

    @staticmethod
    def _proxies_from_clusters(clusters: list[CorrCluster] | None) -> int:
        """Count additional proxy features contributed by correlated clusters."""
        if not clusters:
            return 0
        proxies = 0
        for c in clusters:
            if c.anchor_role == "informative":
                # anchor is informative; remaining (size - 1) columns are proxies
                proxies += max(c.size - 1, 0)
            else:
                # entire cluster is pseudo (latent anchor), so all columns are proxies
                proxies += c.size
        return proxies

    def expected_n_features(self) -> int:
        """Compute n_informative + n_pseudo + n_noise + proxies_from_clusters."""
        proxies = self._proxies_from_clusters(self.corr_clusters)
        return self.n_informative + self.n_pseudo + self.n_noise + proxies

    # @model_validator(mode="after")
    # def _validate_feature_counts(self) -> "DatasetConfig":
    #     """Validate counts and raise a clear error if they are inconsistent."""
    #     # informative anchors must be counted inside n_informative
    #     n_anchors = sum(1 for c in (self.corr_clusters or []) if c.anchor_role == "informative")
    #     if self.n_informative < n_anchors:
    #         raise ValueError(
    #             f"n_informative ({self.n_informative}) < number of informative anchors ({n_anchors})."
    #         )
    #
    #     expected = self.expected_n_features()
    #     if self.n_features != expected:
    #         proxies = self._proxies_from_clusters(self.corr_clusters)
    #         raise ValueError(
    #             "cfg.n_features must equal n_informative + n_pseudo + n_noise + proxies_from_clusters "
    #             f"= {self.n_informative} + {self.n_pseudo} + {self.n_noise} + {proxies} "
    #             f"= {expected}, but got n_features={self.n_features}."
    #         )
    #     return self

    @model_validator(mode="after")
    def _normalize_feature_counts(self) -> DatasetConfig:
        """Ensure counts are consistent; auto-correct n_features with a warning."""
        # Informative anchors must be counted inside n_informative
        n_anchors = sum(1 for c in (self.corr_clusters or []) if c.anchor_role == "informative")
        if self.n_informative < n_anchors:
            raise ValueError(f"n_informative ({self.n_informative}) < number of informative anchors ({n_anchors}).")

        expected = self.expected_n_features()
        if self.n_features != expected:
            warnings.warn(
                "Adjusted n_features to match n_informative + n_pseudo + n_noise + proxies_from_clusters: "
                f"{self.n_informative} + {self.n_pseudo} + {self.n_noise} + "
                f"{self._proxies_from_clusters(self.corr_clusters)} = {expected} "
                f"(was {self.n_features}).",
                UserWarning,
                stacklevel=2,
            )
            # Return a new model with corrected n_features (avoid in-place mutation)
            return self.model_copy(update={"n_features": expected})

        return self

    # --- Convenience helpers for introspection ---------------------------------

    def count_informative_anchors(self) -> int:
        """Count clusters whose anchor contributes as 'informative'."""
        return sum(1 for c in (self.corr_clusters or []) if c.anchor_role == "informative")

    def breakdown(self) -> dict[str, int]:
        """Return a structured breakdown of feature counts, incl. cluster proxies."""
        proxies = self._proxies_from_clusters(self.corr_clusters)
        n_anchors = self.count_informative_anchors()
        return {
            "n_informative_total": int(self.n_informative),
            "n_informative_anchors": int(n_anchors),
            "n_informative_free": int(max(self.n_informative - n_anchors, 0)),
            "n_pseudo_free": int(self.n_pseudo),
            "n_noise": int(self.n_noise),
            "proxies_from_clusters": int(proxies),
            "n_features_expected": int(self.n_informative + self.n_pseudo + self.n_noise + proxies),
            "n_features_configured": int(self.n_features),
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
                "n_pseudo_free",
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
            lines.append(f"- n_pseudo_free          : {b['n_pseudo_free']}")
            lines.append(f"- n_noise                : {b['n_noise']}")
            lines.append(f"- proxies_from_clusters  : {b['proxies_from_clusters']}")
            lines.append(f"- n_features_expected    : {b['n_features_expected']}")
            lines.append(f"- n_features_configured  : {b['n_features_configured']}")

        if per_cluster and self.corr_clusters:
            lines.append("" if as_markdown else "")
            header = (
                "| id | size | role | rho | structure | label | proxies |"
                if as_markdown
                else "Clusters:"
            )
            if as_markdown:
                lines.append("")
                lines.append("### Clusters")
                lines.append("")
                lines.append(header)
                lines.append("|----|------|------|-----|-----------|-------|---------|")
            else:
                lines.append(header)

            for i, c in enumerate(self.corr_clusters, start=1):
                role = c.anchor_role  # expected field in your model
                proxies = (c.size - 1) if role == "informative" else c.size
                label = getattr(c, "label", None)
                if as_markdown:
                    lines.append(
                        f"| {i} | {c.size} | {role} | {c.rho} | {c.structure} | "
                        f"{label if label is not None else ''} | {proxies} |"
                    )
                else:
                    lines.append(
                        f"- #{i}: size={c.size}, role={role}, rho={c.rho}, structure={c.structure}, "
                        f"label={label}, proxies={proxies}"
                    )

        return "\n".join(lines)
