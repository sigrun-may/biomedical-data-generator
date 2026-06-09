# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-06-09

**Breaking release.** The configuration API has been redesigned around the
principle that *relevance is derived from realised signal* rather than declared.
Configs written for the 1.x API will not validate against 2.0.0 and must be
migrated (see the tables below).

### Conceptual change

- **Relevance is derived, not declared.** A correlated cluster no longer carries
  an `anchor_role` flag. Instead a cluster is informative *iff* it expresses a
  signal that varies across classes through one of its optional channels: a
  `mean_channel` (per-class anchor mean shift, first moment) and/or a
  `covariance_channel` (per-class within-cluster correlation, second moment). If
  neither channel varies across classes, the cluster is structural noise.
- **Counts are derived.** `n_informative`, `n_noise`, `n_features`, `n_samples`,
  `n_classes`, and `n_standalone_informative` are now read-only derived
  properties computed from the building blocks. Setting any of them manually is
  rejected.
- **Standalone informative features are configured as groups.** Instead of a
  single `n_informative` count with one global `class_sep`, standalone
  informative features are declared as a list of `StandaloneInformativeGroup`
  entries, each with its own `class_sep`. A list of groups with decreasing
  `class_sep` realises a **signal-strength gradient** across the
  standalone-informative block.

### Migration — `DatasetConfig`

| Old (1.x) | New (2.0) |
| --- | --- |
| `n_informative` (int input) | removed — configure standalone informative features via `standalone_informative_groups` (a list of `StandaloneInformativeGroup`); the informative count is derived |
| `n_noise` (int input) | `n_standalone_noise` |
| `class_sep` (top-level, applied to all informative features) | `StandaloneInformativeGroup.class_sep` (per group) |
| `corr_between` | removed |
| `corr_clusters` | `corr_clusters` (unchanged; cluster fields changed — see below) |

### Migration — `CorrClusterConfig`

| Old (1.x) | New (2.0) |
| --- | --- |
| `correlation` (float, structural) | `baseline_correlation` |
| `correlation` (dict — per-class) | `covariance_channel` → `CovarianceChannel(per_class_correlation=...)` |
| `anchor_effect_size` (`"small"`/`"medium"`/`"large"` or float) | `mean_channel` → `MeanChannel(per_class_effect=...)` |
| `anchor_class` (int) | encoded as the keys of `MeanChannel.per_class_effect` |
| `anchor_role` (`"informative"`/`"noise"`) | removed — relevance is derived from realised signal |
| `structure` | `correlation_structure` |
| `n_cluster_features` | `n_cluster_features` (unchanged) |
| — | `anchor_index` (new; index of the structural anchor within the block) |
| — | `proxy_attenuation` (new; `1.0` reproduces the 1.x anchor-to-proxy propagation) |

### Migration — role index rename (`free_*` → `standalone_*`)

The metadata role partitions that previously used the `free_` prefix now use
`standalone_`, matching the configuration vocabulary:

| Old (1.x) | New (2.0) |
| --- | --- |
| `free_informative_indices` | `standalone_informative_indices` |
| `free_noise_indices` | `standalone_noise_indices` |
