# Quickstart

This short guide shows how to generate a synthetic biomedical dataset using  
**biomedical-data-generator**.  
You will learn:

- how to create a minimal dataset directly in Python,
- how to use YAML configuration files,
- how to inspect dataset metadata,
- how to save the generated dataset.

---

## 1. Installation

```bash
pip install biomedical-data-generator
```

Requires **Python 3.11+**.

---

## 2. Minimal Example (Python only)

The simplest way to generate a dataset is to construct a `DatasetConfig` in Python:

```python
from biomedical_data_generator import (
    DatasetConfig,
    ClassConfig,
    CorrClusterConfig,
    generate_dataset,
)

# Minimal two-class dataset
cfg = DatasetConfig(
    n_informative=5,
    n_noise=3,
    class_configs=[
        ClassConfig(n_samples=60, label="healthy"),
        ClassConfig(n_samples=40, label="diseased"),
    ],
    class_sep=1.2,
    corr_clusters=[
        CorrClusterConfig(
            n_cluster_features=4,
            structure="equicorrelated",
            correlation=0.8,
            anchor_role="informative",
            anchor_effect_size="medium",
            anchor_class=1,
            label="Pathway_A"
        )
    ],
    noise_distribution="normal",
    random_state=42,
)

X, y, meta = generate_dataset(cfg, return_dataframe=True)

print(X.head())
print(y.head())
print(meta)
```

**What this does:**

- Creates a dataset with:  
  - 5 informative features  
  - 3 noise features  
  - one correlated feature cluster with 4 markers  
- Total samples = 60 + 40 = **100**
- Class separation is applied only to informative features
- The correlated cluster anchor is informative and active in class 1

---

## 3. Advanced: YAML Configuration

You can also store the entire configuration in a YAML file  
and load it using `DatasetConfig.from_yaml()`.

### Example: `config.yaml`

```yaml
n_informative: 4
n_noise: 4

class_configs:
  - n_samples: 50
    label: "healthy"

  - n_samples: 50
    label: "diseased"

corr_clusters:
  - n_cluster_features: 5
    structure: "equicorrelated"
    correlation: 0.8
    anchor_role: "informative"
    anchor_effect_size: "medium"
    anchor_class: 1
    label: "Metabolic_Pathway"

  - n_cluster_features: 3
    structure: "toeplitz"
    correlation: 0.4
    anchor_role: "noise"
    label: "Noise_Block"

corr_between: 0.0

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
print(meta)
```

`DatasetConfig.from_yaml()` uses `yaml.safe_load` under the hood and then validates
the raw configuration through the full Pydantic model pipeline. This ensures the
same validation logic as when you build the config in Python.

---

## 4. Saving the Dataset

If you requested a DataFrame (`return_dataframe=True`), saving is straightforward:

```python
X.to_csv("X.csv", index=False)
y.to_csv("y.csv", index=False)
meta.to_yaml("meta.yaml")
```

Or save everything into one Parquet file:

```python
X.to_parquet("dataset.parquet")
```

---

## 5. Inspecting Metadata

`meta` contains the full generative ground truth:

```python
print("Class labels:", meta.class_labels)
print("Batch labels:", getattr(meta, "batch_labels", None))  # only if batch effects are used
print("Correlated clusters:", meta.corr_cluster_indices)
print("Informative features:", meta.informative_features)
```

Metadata fields are designed for:

- benchmarking feature selection,
- sanity-checking structures,
- teaching,
- reproducibility.

---

## 6. Optional: Adding Batch Effects

Batch effects can be added by specifying a `BatchEffectsConfig`:

```python
from biomedical_data_generator import BatchEffectsConfig

cfg.batch_effects = BatchEffectsConfig(
    n_batches=3,
    effect_strength=0.5,
    effect_type="additive",
    effect_granularity="per_feature",
    confounding_with_class=0.3,
    affected_features="all",
)
```
When generating, `meta` will now also include `batch_labels`.
This allows you to benchmark batch correction methods in addition to
classification and feature selection.