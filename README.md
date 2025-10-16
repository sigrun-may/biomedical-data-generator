# Data generator for synthetic data including artificial classes, intraclass correlations, pseudo-classes and random data - [Sphinx Doc](https://sigrun-may.github.io/biomedical-data-generator/)

Generate synthetic classification datasets with correlated feature clusters, controllable class balance, effect sizes, and realistic distractors.

## Highlights

- **Correlated clusters** (equicorrelated or Toeplitz) with one **anchor** feature per cluster
- **Roles**: informative / pseudo / noise
- **Effect size** control via `effect_size` (small/medium/large) and per-cluster `anchor_beta`
- **Multi‑class softmax** generation with `n_classes`, `weights` or exact `class_counts`
- **Feature naming**: tidy prefixes (`i*`, `corr{cid}_k`, `p*`, `n*`) or simple `feature_1..p`
- Returns **(X, y, meta)** with rich `DatasetMeta` (indices, cluster map, class counts, etc.)
- **CLI**: generate from YAML and write CSV in one line

______________________________________________________________________

[![PyPI version](https://badge.fury.io/py/biomedical-data-generator.svg)](https://badge.fury.io/py/biomedical-data-generator)
[![Python Version](https://img.shields.io/pypi/pyversions/biomedical-data-generator.svg)](https://pypi.org/project/biomedical-data-generator/)

## Table of Contents

- [Purpose](#purpose)
- [Data structure](#data-structure)
  - [Different parts of the data set](#different-parts-of-the-data-set)
  - [Data distribution and effect sizes](#data-distribution-and-effect-sizes)
  - [Correlations](#correlations)
- [Pseudo-classes](#pseudo-classes)
- [Random Features](#random-features)
- [Installation](#installation)
- [Licensing](#licensing)

## Purpose

In order to develop new methods or to compare existing methods for feature selection, reference data with known dependencies and importance of the individual features are needed. This data generator can be used to simulate biological data for example artificial high throughput data including artificial biomarkers. Since commonly not all true biomarkers and internal dependencies of high-dimensional biological datasets are known with
certainty, artificial data **enables to know the expected outcome in advance**. In synthetic data, the feature importances and the distribution of each class are known. Irrelevant features can be purely random or belong to a pseudo-class. Such data can be used, for example, to make random effects observable.

- **clear ground truth** (which features truly matter, which are proxies, which are distractors),
- **controllable separability** via `class_sep` and effect sizes,
- **correlated structure** that mimics real omics/tabular data,
- and **reproducible** sampling.
-

## Data structure

### Distributions & effect sizes

- **Effect size** presets (`effect_size ∈ {small, medium, large}`) choose sensible defaults for anchor strengths; you can override per cluster via `anchor_beta`.
- **Class separability** (`class_sep`) scales logits globally; larger → easier classification.
- **Noise controls**: `noise_distribution` (`normal`/`uniform`) and `noise_scale`.

### Correlations

______________________________________________________________________

## Quickstart (Python API)

```python
from biomedical_data_generator import DatasetConfig, generate_dataset

cfg = DatasetConfig(
    n_samples=30,
    n_informative=5,
    n_pseudo=0,
    n_noise=0,
    n_classes=2,
    class_sep=1.5,
    feature_naming="prefixed",
    random_state=42,
)

X, y, meta = generate_dataset(cfg, return_dataframe=True)
print(X.shape, y.shape)        # (30, p), (30,)
print(meta.y_counts)           # e.g. {0: 15, 1: 15}
print(meta.feature_names[:6])  # ['i1', 'i2', ...]
```

**Note on `n_features`:** You typically **omit** it. The generator derives and validates the total as

```
# total feature columns =
# n_informative + n_pseudo + n_noise + (proxies contributed by clusters)
```

If you set `n_features` manually, it must equal that exact sum.

______________________________________________________________________

## Correlated clusters and roles

```python
from biomedical_data_generator import DatasetConfig, CorrCluster, generate_dataset

cfg = DatasetConfig(
    n_samples=200,
    n_informative=4,
    n_pseudo=1,
    n_noise=3,
    n_classes=3,
    class_sep=1.2,
    corr_clusters=[
        CorrCluster(size=3, rho=0.7, structure="equicorrelated",
                    anchor_role="informative", anchor_beta=1.0),
        CorrCluster(size=2, rho=0.6, structure="toeplitz",
                    anchor_role="pseudo"),
    ],
    random_state=0,
)

X, y, meta = generate_dataset(cfg, return_dataframe=True)
```

**How it works (overview):**

- Each cluster contributes `size` columns: **1 anchor** + `(size-1)` **proxies**.

- `anchor_role` controls the anchor’s contribution to the label:

  - `informative`: contributes to a target class (via `anchor_beta`).
  - `pseudo`: correlated but **β = 0** (distractor).
  - `noise`: uninformative and uncorrelated with the label; proxies act as pseudo.

- `structure` ∈ {`equicorrelated`, `toeplitz`} defines within‑cluster correlations.

- Global `effect_size` = {`small`, `medium`, `large`} sets sensible defaults for `anchor_beta`.

- `anchor_mode` = {`equalized`, `strong`} tunes how anchors are distributed/weighted across classes.

**Feature ordering** in `X`:

1. Cluster features (anchors first **within** each cluster)
1. Free informative features (`i*`)
1. Free pseudo features (`p*`)
1. Noise features (`n*`)

______________________________________________________________________

## Class balance and separability

- `n_classes`: number of classes (≥ 2)
- `weights`: relative class priors (normalized internally)
- `class_counts`: **exact** per‑class sizes (overrides `weights`)
- `class_sep`: scales logits → higher means easier separation

Under the hood, labels come from a **softmax** model. Cluster anchors add to their target class; free
informative features contribute in a round‑robin fashion (β≈0.8). Pseudo & noise have β=0.

______________________________________________________________________

## Naming and convenience

- `feature_naming`: `"prefixed"` (default) or `"simple"`

- Prefixes (when `prefixed`):

  - Informative: `i` → `i1, i2, …`
  - Pseudo: `p` → `p1, p2, …`
  - Noise: `n` → `n1, n2, …`
  - Correlated cluster proxies: `corr{cid}_{k}`

Reproducibility via `random_state` (global), plus optional per‑cluster seeds.

______________________________________________________________________

## Return values

`generate_dataset(cfg, return_dataframe=True)` → `(X, y, meta)`

- **X**: `pandas.DataFrame` (or `np.ndarray` if `return_dataframe=False`)

- **y**: `np.ndarray[int]` labels

- **meta**: `DatasetMeta` with (selected):

  - `feature_names`
  - `informative_idx`, `pseudo_idx`, `noise_idx`
  - `corr_cluster_indices: dict[int, list[int]]`
  - `anchor_idx: dict[int, int | None]`
  - `anchor_role: dict[int, str]`
  - `anchor_beta: dict[int, float]`
  - `y_weights: tuple[float, …]`, `y_counts: dict[int, int]`

______________________________________________________________________

## Command‑line usage

The package ships a small CLI named **`bdg`**.

```bash
# Generate from YAML and write a single CSV with features + class
bdg --config config.yaml --out dataset.csv

# Print only metadata (JSON) to stdout
bdg --config config.yaml
```

**Minimal YAML example**

```yaml
n_samples: 200
n_classes: 3
class_sep: 1.2
n_informative: 4
n_pseudo: 1
n_noise: 3
feature_naming: prefixed
corr_clusters:
  - size: 3
    rho: 0.7
    structure: equicorrelated
    anchor_role: informative
    anchor_beta: 1.0
  - size: 2
    rho: 0.6
    structure: toeplitz
    anchor_role: pseudo
random_state: 0
```

The CLI prints `DatasetMeta` as JSON to stdout. When `--out` is given, it also writes `dataset.csv`
with one row per sample and a final `class` column.

______________________________________________________________________

## API reference (essentials)

```python
from biomedical_data_generator import (
    DatasetConfig, CorrCluster, NoiseDistribution,
    generate_dataset, generate_correlated_cluster,
    find_seed_for_correlation, DatasetMeta,
)
```

- `generate_dataset(cfg, return_dataframe=True, **overrides)` → `(X, y, meta)`
- `generate_correlated_cluster(n_samples, size, rho, structure, random_state=None)` → `(Xc, meta)`
- `find_seed_for_correlation(cfg, target_rho, kind="pearson", mode="min|max", max_tries=1000, threshold=None)` → best seed and dataset tuple

______________________________________________________________________

## Tips

- Prefer **omitting `n_features`**; let the config derive a consistent total.
- Use `class_counts` when you need exact per‑class sizes (it overrides `weights`).
- Start with `effect_size="medium"` and `class_sep≈1.2–1.5`; then tune for your lesson/demo.
- Keep `feature_naming="prefixed"` for didactics — it makes plots and tables easier to read.

______________________________________________________________________

## License

MIT © Sigrun May

______________________________________________________________________

## Acknowledgments

Developed for teaching synthetic biomedical data generation and ML workflows.

### Different parts of the data set

The biomedical-data-generator produces data sets consisting of up to three main parts:

1. **Relevant/ informative features** belonging to an artificial class (for example artificial biomarkers)
1. [optional] **Pseudo-classes** (for example a patient's height or gender, which have no association with a particular disease)
1. [optional] **Random data** representing the features (for example biomarker candidates) that are not associated with any class. This can be used to simulate random effects that occur in small sample sizes with a very large number of features. Or noise that occurs in real data.

The number of artificial classes is not limited. Each class is generated individually and then combined with the others.
In order to simulate artificial biomarkers in total, all individual classes have the same number of features in total.

This is an example of simulated binary biological data including artificial biomarkers:

![Different blocks of the artificial data.](docs/source/imgs/artificial_data.png)

- **Informative features** (`i*`): truly predictive; include **cluster anchors** if `anchor_role="informative"`.
- **Pseudo features** (`p*`): belong to a pseudo-class, thus non‑causal but misleading.
- **Noise features** (`n*`): random, uncorrelated with the label; useful to test robustness.
- **Correlated clusters** (`corr{cid}_k`): within a cluster, one **anchor** + `(size-1)` \*\*proxies`; correlation structure `equicorrelated`or`toeplitz\`.

### Data distribution and effect sizes

For each class, either the **normal distribution or the log normal distribution** can be selected. The different **classes can be shifted** to regulate the effect sizes and to influence the difficulty of data analysis.

The normally distributed data could, for example, represent the range of values of healthy individuals.
In the case of a disease, biological systems are in some way out of balance.
Extreme changes in values as well as outliers can then be observed ([Concordet et al., 2009](https://doi.org/10.1016/j.cca.2009.03.057)).
Therefore, the values of a diseased individual could be simulated with a lognormal distribution.

Example of log-normal and normal distributed classes:

![Different distributions of the classes.](docs/source/imgs/distributions.png)

### Correlations

**Intra-class correlation can be generated for each artificial class**. Any number of groups
containing correlated features can be combined with any given number of uncorrelated features.

However, a high correlation within a group does not necessarily lead to
a high correlation to other groups or features of the same class. An example of a class with three
highly correlated groups but without high correlations between all groups:

![Different distributions of the classes.](docs/source/imgs/corr_3_groups.png)

It is probably likely that biomarkers of healthy individuals usually have a relatively low correlation. On average,
their values are within a usual "normal" range. In this case, one biomarker tends to be in the upper normal range and another biomarker in the lower normal range. However, individually it can also be exactly the opposite, so that the correlation between healthy individuals would be rather low. Therefore, the **values of healthy people
could be simulated without any special artificially generated correlations**.

In the case of a disease, however, a biological system is brought out of balance in a certain way and must react to it.
For example, this reaction can then happen in a coordinated manner involving several biomarkers,
or corresponding cascades (e.g. pathways) can be activated or blocked. This can result in a **rather stronger
correlation of biomarkers in patients suffering from a disease**. To simulate these intra-class correlations,
a class is divided into a given number of groups with high internal correlation
(the respective strength can be defined).

- **Equicorrelated**: roughly constant pairwise correlation within a cluster.
- **Toeplitz (AR‑like)**: correlation decays with feature distance (`rho**|i−j|`).

## Pseudo-classes

One option for an element of the generated data set is a pseudo-class. For example, this could be a
patient's height or gender, which are not related to a specific disease.

The generated pseudo-class contains the same number of classes with identical distributions as the artificial biomarkers.
But after the generation of the individual classes, all samples (rows) are randomly shuffled.
Finally, combining the shuffled data with the original, unshuffled class labels, the pseudo-class no longer
has a valid association with any class label. Consequently, no element of the pseudo-class should be
recognized as relevant by a feature selection algorithm.

## Random Features

The artificial biomarkers and, if applicable, the optional pseudo-classes can be combined with any number
of random features. Varying the number of random features can be used, for example, to analyze random effects
that occur in small sample sizes with a very large number of features.

## Installation

The biomedical-data-generator is available at [the Python Package Index (PyPI)](https://pypi.org/project/biomedical-data-generator/).
It can be installed with pip:

```bash
$ pip install biomedical-data-generator
```

> Tested with Python **3.11+**.

## Quickstart example

This is an example of how to generate a synthetic dataset:

```python
from biomedical_data_generator import DatasetConfig, generate_dataset

cfg = DatasetConfig(
    n_samples=30,
    n_informative=5,
    n_pseudo=0,
    n_noise=0,
    n_classes=2,
    class_sep=1.0,
    feature_naming="prefixed",
    random_state=42,
)

X, y, meta = generate_dataset(cfg, return_dataframe=True)
print(X.shape, y.shape, meta.y_counts)
```

## Licensing

Copyright (c) 2025 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften

Licensed under the **MIT License** (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License by reviewing the file
[LICENSE](https://github.com/sigrun-may/biomedical-data-generator/blob/main/LICENSE) in the repository.

## Quickstart (Python API)

```python
from biomedical_data_generator import DatasetConfig, generate_dataset

cfg = DatasetConfig(
    n_samples=30,
    n_informative=5,
    n_pseudo=0,
    n_noise=0,
    n_classes=2,
    class_sep=1.5,
    feature_naming="prefixed",
    random_state=42,
)

X, y, meta = generate_dataset(cfg, return_dataframe=True)
print(X.shape, y.shape)        # (30, p), (30,)
print(meta.y_counts)           # e.g. {0: 15, 1: 15}
print(meta.feature_names[:6])  # ['i1', 'i2', ...]
```

**Note on `n_features`:** You typically **omit** it. The generator derives and validates the total as

```
# total feature columns =
# n_informative + n_pseudo + n_noise + (proxies contributed by clusters)
```

If you set `n_features` manually, it must equal that exact sum.

______________________________________________________________________

## Correlated clusters and roles

```python
from biomedical_data_generator import DatasetConfig, CorrCluster, generate_dataset

cfg = DatasetConfig(
    n_samples=200,
    n_informative=4,
    n_pseudo=1,
    n_noise=3,
    n_classes=3,
    class_sep=1.2,
    corr_clusters=[
        CorrCluster(size=3, rho=0.7, structure="equicorrelated",
                    anchor_role="informative", anchor_beta=1.0),
        CorrCluster(size=2, rho=0.6, structure="toeplitz",
                    anchor_role="pseudo"),
    ],
    random_state=0,
)

X, y, meta = generate_dataset(cfg, return_dataframe=True)
```

**How it works (overview):**

- Each cluster contributes `size` columns: **1 anchor** + `(size-1)` **proxies**.

- `anchor_role` controls the anchor’s contribution to the label:

  - `informative`: contributes to a target class (via `anchor_beta`).
  - `pseudo`: correlated but **β = 0** (distractor).
  - `noise`: uninformative and uncorrelated with the label; proxies act as pseudo.

- `structure` ∈ {`equicorrelated`, `toeplitz`} defines within‑cluster correlations.

- Global `effect_size` = {`small`, `medium`, `large`} sets sensible defaults for `anchor_beta`.

- `anchor_mode` = {`equalized`, `strong`} tunes how anchors are distributed/weighted across classes.

**Feature ordering** in `X`:

1. Cluster features (anchors first **within** each cluster)
1. Free informative features (`i*`)
1. Free pseudo features (`p*`)
1. Noise features (`n*`)

______________________________________________________________________

## Class balance and separability

- `n_classes`: number of classes (≥ 2)
- `weights`: relative class priors (normalized internally)
- `class_counts`: **exact** per‑class sizes (overrides `weights`)
- `class_sep`: scales logits → higher means easier separation

Under the hood, labels come from a **softmax** model. Cluster anchors add to their target class; free
informative features contribute in a round‑robin fashion (β≈0.8). Pseudo & noise have β=0.

______________________________________________________________________

## Naming and convenience

- `feature_naming`: `"prefixed"` (default) or `"simple"`

- Prefixes (when `prefixed`):

  - Informative: `i` → `i1, i2, …`
  - Pseudo: `p` → `p1, p2, …`
  - Noise: `n` → `n1, n2, …`
  - Correlated cluster proxies: `corr{cid}_{k}`

Reproducibility via `random_state` (global), plus optional per‑cluster seeds.

______________________________________________________________________

## Return values

`generate_dataset(cfg, return_dataframe=True)` → `(X, y, meta)`

- **X**: `pandas.DataFrame` (or `np.ndarray` if `return_dataframe=False`)

- **y**: `np.ndarray[int]` labels

- **meta**: `DatasetMeta` with (selected):

  - `feature_names`
  - `informative_idx`, `pseudo_idx`, `noise_idx`
  - `corr_cluster_indices: dict[int, list[int]]`
  - `anchor_idx: dict[int, int | None]`
  - `anchor_role: dict[int, str]`
  - `anchor_beta: dict[int, float]`
  - `y_weights: tuple[float, …]`, `y_counts: dict[int, int]`

______________________________________________________________________

## Command‑line usage

The package ships a small CLI named **`bdg`**.

```bash
# Generate from YAML and write a single CSV with features + class
bdg --config config.yaml --out dataset.csv

# Print only metadata (JSON) to stdout
bdg --config config.yaml
```

**Minimal YAML example**

```yaml
n_samples: 200
n_classes: 3
class_sep: 1.2
n_informative: 4
n_pseudo: 1
n_noise: 3
feature_naming: prefixed
corr_clusters:
  - size: 3
    rho: 0.7
    structure: equicorrelated
    anchor_role: informative
    anchor_beta: 1.0
  - size: 2
    rho: 0.6
    structure: toeplitz
    anchor_role: pseudo
random_state: 0
```

The CLI prints `DatasetMeta` as JSON to stdout. When `--out` is given, it also writes `dataset.csv`
with one row per sample and a final `class` column.

______________________________________________________________________

## API reference (essentials)

```python
from biomedical_data_generator import (
    DatasetConfig, CorrCluster, NoiseDistribution,
    generate_dataset, generate_correlated_cluster,
    find_seed_for_correlation, DatasetMeta,
)
```

- `generate_dataset(cfg, return_dataframe=True, **overrides)` → `(X, y, meta)`
- `generate_correlated_cluster(n_samples, size, rho, structure, random_state=None)` → `(Xc, meta)`
- `find_seed_for_correlation(cfg, target_rho, kind="pearson", mode="min|max", max_tries=1000, threshold=None)` → best seed and dataset tuple

______________________________________________________________________

## Tips

- Prefer **omitting `n_features`**; let the config derive a consistent total.
- Use `class_counts` when you need exact per‑class sizes (it overrides `weights`).
- Start with `effect_size="medium"` and `class_sep≈1.2–1.5`; then tune for your lesson/demo.
- Keep `feature_naming="prefixed"` for didactics — it makes plots and tables easier to read.

______________________________________________________________________
