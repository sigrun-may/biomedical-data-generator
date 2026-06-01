# Copyright (c) 2026 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Feature-role taxonomy and Kohavi-John relevance interpretation.

This module provides a two-layer reading of a generated dataset's ground truth:

Layer 1 (generative roles)
    A purely structural partition of the feature columns into the six
    roles that the generator distinguishes. The six roles arise from two
    orthogonal distinctions.

    The first concerns whether a feature carries class-discriminating signal:
    *informative* features encode a mean shift between classes, whereas
    *noise* features do not.

    The second concerns how a feature relates to the correlated feature clusters,
    which are optional. When no cluster is configured, every feature is
    *free* -- independent and belonging to no cluster -- so the dataset
    contains nothing but free informative and free noise features. When a
    cluster is generated, its lead feature is the *anchor*; if its
    ``anchor_role`` is informative, the anchor is the only cluster column to
    which a mean shift is applied directly. Each remaining cluster member is a
    *proxy*: it is correlated with its anchor and therefore inherits an
    attenuated version of the anchor's behaviour.

    Combining the two distinctions yields the six roles:

    * **free informative** -- an independent informative feature outside
      any cluster. It carries a class-separating mean shift on its own and
      is not part of any generated correlation structure.
    * **informative anchor** -- the lead feature of a cluster whose
      ``anchor_role="informative"``. It receives the class-specific mean
      shift and simultaneously seeds the within-cluster correlation
      shared by its proxies.
    * **informative proxy** -- a non-anchor member of an informative
      cluster. It carries no shift of its own but inherits an attenuated
      signal through its correlation with the informative anchor.
    * **free noise** -- an independent noise feature outside any cluster.
      It carries neither dedicated class signal nor correlation structure.
    * **noise anchor** -- the lead feature of a cluster whose
      ``anchor_role="noise"``. It receives no mean shift but still seeds a
      within-cluster correlation structure shared with its proxies.
    * **noise proxy** -- a non-anchor member of a noise cluster. It is
      correlated with the noise anchor and forms a purely structural,
      signal-free correlated block.

    This layer is exact and assumption-free: it is derived only from the
    index bookkeeping already stored on
    :class:`~biomedical_data_generator.meta.DatasetMeta`.

Layer 2 (Kohavi-John relevance)
    An interpretation of the generative roles in terms of the
    Kohavi & John (1997) relevance categories (strongly relevant, weakly
    relevant, irrelevant). This mapping is exact only when the generative
    design keeps the categories cleanly separated; otherwise it is an
    approximation and the returned :class:`RelevanceView` flags this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from biomedical_data_generator.meta import DatasetMeta

__all__ = [
    "FeatureRoles",
    "RelevanceView",
    "compute_feature_roles",
    "compute_relevance",
]

# Correlations whose magnitude is below this threshold are treated as zero.
_CORRELATION_ZERO_THRESHOLD = 1e-12


@dataclass(frozen=True)
class FeatureRoles:
    """Generative feature roles derived from a DatasetMeta (Layer 1, exact).

    Attributes:
        free_informative_indices: Informative features that live outside any
            correlated cluster (each carries a unique class signal).
        informative_anchor_indices: Anchor columns of clusters whose anchor
            role is ``"informative"``.
        informative_proxy_indices: Proxy columns of informative clusters
            (they inherit an attenuated, correlated copy of the anchor signal).
        free_noise_indices: Independent noise features outside any cluster.
        noise_anchor_indices: Anchor columns of clusters whose anchor role is
            ``"noise"``.
        noise_proxy_indices: Proxy columns of noise clusters.
        cluster_membership: Mapping ``column_index -> cluster_id`` for every
            column that belongs to a correlated cluster.
    """

    free_informative_indices: list[int]
    informative_anchor_indices: list[int]
    informative_proxy_indices: list[int]
    free_noise_indices: list[int]
    noise_anchor_indices: list[int]
    noise_proxy_indices: list[int]
    cluster_membership: dict[int, int]


@dataclass(frozen=True)
class RelevanceView:
    """Kohavi-John relevance interpretation (Layer 2, assumption-guarded).

    Attributes:
        strongly_relevant_indices: Features carrying unique class information
            whose removal degrades the optimal classifier.
        weakly_relevant_indices: Features carrying class information that is
            redundantly available through correlated partners.
        irrelevant_indices: Features carrying no class information.
        mapping_is_exact: ``True`` when the generative design guarantees the
            mapping (mutually independent clusters and class-independent
            correlations); ``False`` when cross-role correlations or
            class-specific correlations make it an approximation.
        assumptions: Human-readable note explaining why the mapping is exact
            or approximate.
    """

    strongly_relevant_indices: list[int]
    weakly_relevant_indices: list[int]
    irrelevant_indices: list[int]
    mapping_is_exact: bool
    assumptions: str  # human-readable note on why exact / approximate


def compute_feature_roles(meta: DatasetMeta) -> FeatureRoles:
    """Derive the six generative roles. Pure function, no assumptions.

    The partition is reconstructed from the index bookkeeping stored on
    ``meta``:

    * ``meta.informative_idx`` holds free informative columns plus informative
      anchors; ``meta.noise_idx`` holds free noise columns only.
    * Cluster layouts come from ``meta.corr_cluster_indices`` and
      ``meta.anchor_idx``; the per-cluster anchor role from ``meta.anchor_role``.

    Anchor and proxy columns (and noise anchors in particular) are therefore
    recovered from the cluster layout rather than from the flat index lists.

    Args:
        meta: Resolved ground-truth metadata of a generated dataset.

    Returns:
        A :class:`FeatureRoles` instance partitioning the columns into the six
        generative roles plus the cluster-membership map.
    """
    informative_anchor_indices: list[int] = []
    informative_proxy_indices: list[int] = []
    noise_anchor_indices: list[int] = []
    noise_proxy_indices: list[int] = []
    cluster_membership: dict[int, int] = {}

    for cluster_id, column_indices in meta.corr_cluster_indices.items():
        anchor_column = meta.anchor_idx[cluster_id]
        anchor_role = meta.anchor_role[cluster_id]

        for column_index in column_indices:
            cluster_membership[column_index] = cluster_id

        proxy_columns = [column for column in column_indices if column != anchor_column]

        if anchor_role == "informative":
            informative_anchor_indices.append(anchor_column)
            informative_proxy_indices.extend(proxy_columns)
        else:
            noise_anchor_indices.append(anchor_column)
            noise_proxy_indices.extend(proxy_columns)

    # Free informative features are the informative columns that are not anchors.
    informative_anchor_set = set(informative_anchor_indices)
    free_informative_indices = [
        column for column in meta.informative_idx if column not in informative_anchor_set
    ]

    # meta.noise_idx already contains only the independent (free) noise columns.
    free_noise_indices = list(meta.noise_idx)

    return FeatureRoles(
        free_informative_indices=sorted(free_informative_indices),
        informative_anchor_indices=sorted(informative_anchor_indices),
        informative_proxy_indices=sorted(informative_proxy_indices),
        free_noise_indices=sorted(free_noise_indices),
        noise_anchor_indices=sorted(noise_anchor_indices),
        noise_proxy_indices=sorted(noise_proxy_indices),
        cluster_membership=cluster_membership,
    )


def compute_relevance(meta: DatasetMeta) -> RelevanceView:
    """Map generative roles onto Kohavi-John relevance categories.

    Exact only when clusters are mutually independent (corr_between == 0)
    and no class-specific correlations are used; otherwise approximate.

    The mapping follows the generative design intent:

    * Free informative features carry a unique class signal that no other
      feature replicates, so they are **strongly relevant**.
    * An informative cluster is a redundant module: anchor and proxies all
      carry the same class signal and substitute for one another, so each
      member is **weakly relevant**. A degenerate informative cluster with no
      proxies has a single, non-redundant member, so that anchor is
      **strongly relevant**.
    * Noise features (free noise, noise anchors and noise proxies) carry no
      class signal and are therefore **irrelevant**.

    Args:
        meta: Resolved ground-truth metadata of a generated dataset.

    Returns:
        A :class:`RelevanceView` with the three relevance partitions, the
        ``mapping_is_exact`` flag and a human-readable assumptions note.
    """
    roles = compute_feature_roles(meta)

    # Cluster sizes let us separate redundant informative anchors (clusters
    # with at least one proxy) from singleton informative anchors (no proxy).
    redundant_informative_anchor_indices: list[int] = []
    singleton_informative_anchor_indices: list[int] = []
    for anchor_column in roles.informative_anchor_indices:
        cluster_id = roles.cluster_membership[anchor_column]
        cluster_size = len(meta.corr_cluster_indices[cluster_id])
        if cluster_size > 1:
            redundant_informative_anchor_indices.append(anchor_column)
        else:
            singleton_informative_anchor_indices.append(anchor_column)

    strongly_relevant_indices = sorted(
        roles.free_informative_indices + singleton_informative_anchor_indices
    )
    weakly_relevant_indices = sorted(
        redundant_informative_anchor_indices + roles.informative_proxy_indices
    )
    irrelevant_indices = sorted(
        roles.free_noise_indices + roles.noise_anchor_indices + roles.noise_proxy_indices
    )

    clusters_are_independent = abs(float(meta.corr_between)) < _CORRELATION_ZERO_THRESHOLD
    has_class_specific_correlation = any(
        isinstance(correlation, dict) for correlation in meta.cluster_correlation.values()
    )
    mapping_is_exact = clusters_are_independent and not has_class_specific_correlation

    assumptions = _describe_assumptions(
        clusters_are_independent=clusters_are_independent,
        has_class_specific_correlation=has_class_specific_correlation,
    )

    return RelevanceView(
        strongly_relevant_indices=strongly_relevant_indices,
        weakly_relevant_indices=weakly_relevant_indices,
        irrelevant_indices=irrelevant_indices,
        mapping_is_exact=mapping_is_exact,
        assumptions=assumptions,
    )


def _describe_assumptions(
    *,
    clusters_are_independent: bool,
    has_class_specific_correlation: bool,
) -> str:
    """Build the human-readable assumptions note for a :class:`RelevanceView`.

    Args:
        clusters_are_independent: Whether ``corr_between`` is effectively zero.
        has_class_specific_correlation: Whether any cluster uses a per-class
            correlation mapping.

    Returns:
        A note stating why the relevance mapping is exact or approximate.
    """
    if clusters_are_independent and not has_class_specific_correlation:
        return (
            "Exact mapping: clusters are mutually independent (corr_between == 0) "
            "and all correlations are global (class-independent), so the generative "
            "roles map exactly onto the Kohavi-John relevance categories."
        )

    reasons: list[str] = []
    if not clusters_are_independent:
        reasons.append(
            "corr_between != 0 introduces correlations across role boundaries, so "
            "free informative and noise features are no longer cleanly separated"
        )
    if has_class_specific_correlation:
        reasons.append(
            "class-specific correlations make the correlation structure itself "
            "class-informative, so some noise proxies may become weakly relevant"
        )
    return "Approximate mapping: " + "; ".join(reasons) + "."
