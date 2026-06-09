# Quickstart

This short guide shows how to generate a synthetic biomedical dataset using
**biomedical-data-generator**.
You will learn:

- how to create a dataset with the channel-based API directly in Python,
- how signal is expressed structurally (a separation gradient and correlated clusters),
- how to read the derived ground truth (`FeatureRoles` and `FeatureStrengths`),
- how to use YAML configuration files,
- how to add batch effects with controlled class confounding,
- how to save the generated dataset.

The package targets biomedical settings, where data are typically high-dimensional
(many features, comparatively few samples). What it gives you that a generic
`make_classification` does not is **role-aware ground truth** — every column is
traceable to the mechanism that generated it — and **explicit class–batch
confounding** for studying non-causal variation.

---

## 1. Installation

```bash
pip install biomedical-data-generator
```

Requires **Python 3.10+**.

---

## 2. Minimal Example (Python only)

Signal is never *declared*; it is expressed through structure. Standalone
informative features live in `StandaloneInformativeGroup` blocks, each carrying its
own `class_sep` (per-class mean separation). Correlated clusters are structural by
default and become *informative* only when a channel varies across classes: a
`MeanChannel` (first-moment shift on the anchor) or a `CovarianceChannel`
(second-moment / within-cluster correlation). Relevance is therefore *derived* from
the channels, never set by hand.

```python
from biomedical_data_generator import (
    DatasetConfig,
    ClassConfig,
    CorrClusterConfig,
    MeanChannel,
    CovarianceChannel,
    StandaloneInformativeGroup,
    generate_dataset,
)

cfg = DatasetConfig(
    # A signal-strength gradient: three groups with decreasing separation.
    standalone_informative_groups=[
        StandaloneInformativeGroup(n_features=3, class_sep=2.0),  # strong
        StandaloneInformativeGroup(n_features=3, class_sep=1.0),  # medium
        StandaloneInformativeGroup(n_features=3, class_sep=0.4),  # weak
    ],
    # Independent noise features with no class signal.
    n_standalone_noise=4,
    class_configs=[
        ClassConfig(n_samples=60, label="healthy"),
        ClassConfig(n_samples=40, label="diseased"),
    ],
    corr_clusters=[
        # Informative via a mean shift on the diseased class (first moment).
        CorrClusterConfig(
            n_cluster_features=4,
            correlation_structure="equicorrelated",
            baseline_correlation=0.6,
            mean_channel=MeanChannel(per_class_effect={1: 1.5}),
            label="Pathway_A (mean shift in diseased)",
        ),
        # Informative via differential co-expression (second moment): the
        # within-cluster correlation differs between classes.
        CorrClusterConfig(
            n_cluster_features=3,
            correlation_structure="equicorrelated",
            baseline_correlation=0.3,
            covariance_channel=CovarianceChannel(per_class_correlation={0: 0.1, 1: 0.8}),
            label="Pathway_B (differential co-expression)",
        ),
    ],
    noise_distribution="normal",
    random_state=42,
)

X, y, meta = generate_dataset(cfg, return_dataframe=True)

print(X.shape, y.shape)        # (100, 20) (100,)
print(X.columns.tolist())
print(cfg.breakdown())
```

**What this does:**

- Builds a **separation gradient**: 9 standalone informative features split into
  strong / medium / weak groups (`class_sep` 2.0 / 1.0 / 0.4).
- Adds 4 standalone noise features.
- Adds two correlated clusters, each made informative through a *different* channel:
  `Pathway_A` through a class-dependent mean shift, `Pathway_B` through a
  class-dependent within-cluster correlation.
- Total samples = 60 + 40 = **100**; total features = **20**.

`cfg.breakdown()` reports the derived counts (informative vs. noise are *derived*
from the channels, never set directly):

```python
{'n_standalone_informative': 9, 'n_standalone_noise': 4, 'n_cluster_members': 7,
 'n_informative': 16, 'n_noise': 4, 'n_features': 20}
```

---

## 3. Reading the Ground Truth

The distinguishing feature of this generator is that every column is traceable to
the mechanism that produced it. Two pure functions derive that ground truth from
`meta` — no feature matrix required.

`compute_feature_roles` returns the structural six-way partition of the columns:

```python
from biomedical_data_generator import compute_feature_roles

roles = compute_feature_roles(meta)
print("standalone informative:", roles.standalone_informative_indices)  # [0..8]
print("informative anchors:   ", roles.informative_anchor_indices)       # [9, 13]
print("informative proxies:   ", roles.informative_proxy_indices)        # [10, 11, 12, 14, 15]
print("standalone noise:      ", roles.standalone_noise_indices)         # [16, 17, 18, 19]
print("noise anchors:         ", roles.noise_anchor_indices)             # []
print("noise proxies:         ", roles.noise_proxy_indices)              # []
```

A cluster anchor is the only column shifted directly; each proxy inherits an
attenuated version of the anchor's signal through correlation. Because both
clusters here carry signal, there are no noise anchors or proxies.

`compute_feature_strengths` returns the per-feature signal strengths and the set of
active channels per feature:

```python
from biomedical_data_generator import compute_feature_strengths

strengths = compute_feature_strengths(meta)
for name, channels, m, c in zip(
    meta.feature_names,
    strengths.signal_channels,
    strengths.mean_strength,
    strengths.covariance_strength,
):
    print(f"{name:>16}  channels={channels!s:<22} mean={m:.2f} cov={c:.2f}")
```

The gradient and the two channel types are visible directly in the output:

```text
              i1  channels=('mean',)            mean=2.00 cov=0.00
              i4  channels=('mean',)            mean=1.00 cov=0.00
              i7  channels=('mean',)            mean=0.40 cov=0.00
    corr1_anchor  channels=('mean',)            mean=1.50 cov=0.00
         corr1_2  channels=('mean',)            mean=0.90 cov=0.00
    corr2_anchor  channels=('covariance',)      mean=0.00 cov=0.70
              n1  channels=()                   mean=0.00 cov=0.00
```

`mean_strength` follows the gradient (2.0 → 1.0 → 0.4); proxies of `Pathway_A`
inherit an attenuated mean shift (0.9 = 1.5 × 0.6); `Pathway_B` carries a pure
covariance signal; noise features have empty channels.

---

## 4. Advanced: YAML Configuration

You can store the entire configuration in a YAML file and load it with
`DatasetConfig.from_yaml()`. Channels are nested mappings.

### Example: `config.yaml`

```yaml
standalone_informative_groups:
  - n_features: 4
    class_sep: 1.5
  - n_features: 4
    class_sep: 0.5

n_standalone_noise: 4

class_configs:
  - n_samples: 50
    label: "healthy"
  - n_samples: 50
    label: "diseased"

corr_clusters:
  - n_cluster_features: 5
    correlation_structure: "equicorrelated"
    baseline_correlation: 0.6
    mean_channel:
      per_class_effect:
        1: 1.5
    label: "Metabolic_Pathway"

  # No channel varies across classes → derived as a purely structural (noise) block.
  - n_cluster_features: 3
    correlation_structure: "toeplitz"
    baseline_correlation: 0.4
    label: "Structural_Block"

noise_distribution: "normal"
noise_distribution_params:
  loc: 0
  scale: 1

prefixed_feature_naming: true
random_state: 123
```

### Load & generate dataset:

```python
from biomedical_data_generator import DatasetConfig, generate_dataset

cfg = DatasetConfig.from_yaml("config.yaml")
X, y, meta = generate_dataset(cfg, return_dataframe=True)

print(cfg.breakdown())
print(cfg.cluster_informative_flags())  # [True, False]
```

`DatasetConfig.from_yaml()` uses `yaml.safe_load` under the hood and then validates
the raw configuration through the full Pydantic model pipeline. This ensures the
same validation logic as when you build the config in Python. Note that
`Structural_Block` defines no class-varying channel, so it is *derived* as noise —
`cluster_informative_flags()` returns `[True, False]`.

---

## 5. Saving the Dataset

If you requested a DataFrame (`return_dataframe=True`), saving is straightforward.
`y` is a NumPy array, so wrap it before writing to CSV:

```python
import pandas as pd

X.to_csv("X.csv", index=False)
pd.Series(y, name="label").to_csv("y.csv", index=False)
```

The complete ground truth is serializable via `meta.to_dict()`:

```python
import json

with open("meta.json", "w") as f:
    json.dump(meta.to_dict(), f, indent=2)
```

Or save the feature matrix into one Parquet file:

```python
X.to_parquet("dataset.parquet")
```

---

## 6. Inspecting Metadata

`meta` carries the full generative ground truth:

```python
print("Class names:", meta.class_names)              # ['healthy', 'diseased']
print("Samples per class:", meta.samples_per_class)  # {0: 60, 1: 40}
print("Correlated clusters:", meta.corr_cluster_indices)  # {0: [9, 10, 11, 12], 1: [13, 14, 15]}
print("Informative columns:", meta.informative_idx)
print("Noise columns:", meta.noise_idx)
print("Batch labels:", meta.batch_labels)  # None unless batch effects are configured
```

This ground truth is designed for:

- benchmarking feature selection and importance metrics against known roles,
- separating first-moment (mean) from second-moment (correlation) signal,
- studying the effect of correlated proxies on interpretability,
- reproducibility and teaching.

---

## 7. Optional: Batch Effects and Class Confounding

Batch effects add a technical overlay on top of the biological signal. The key
control is `confounding_with_class`, which couples batch assignment to the class
label — letting you study how non-causal variation can be mistaken for signal:

```python
from biomedical_data_generator import BatchEffectsConfig

cfg.batch_effects = BatchEffectsConfig(
    n_batches=3,
    effect_strength=0.5,
    effect_type="additive",
    effect_granularity="per_feature",
    confounding_with_class=0.6,  # batch correlates with class (recruitment bias)
    affected_features="all",
)

X, y, meta = generate_dataset(cfg, return_dataframe=True)
print(meta.batch_labels[:10])  # per-sample batch IDs
```

When batch effects are applied, `meta.batch` (and the `meta.batch_labels`
accessor) record the per-sample batch assignments and the applied effects. With
`confounding_with_class > 0`, class and batch are deliberately entangled — the
setting for benchmarking batch-correction methods and exposing models that latch
onto batch rather than biology.
