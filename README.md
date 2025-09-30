# Data generator for synthetic data including artificial classes, intraclass correlations, pseudo-classes and random data - [Sphinx Doc](https://sigrun-may.github.io/biomedical-data-generator/)

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

## Data structure

### Different parts of the data set

The biomedical-data-generator produces data sets consisting of up to three main parts:

1. **Relevant features** belonging to an artificial class (for example artificial biomarkers)
1. [optional] **Pseudo-classes** (for example a patient's height or gender, which have no association with a particular disease)
1. [optional] **Random data** representing the features (for example biomarker candidates) that are not associated with any class

The number of artificial classes is not limited. Each class is generated individually and then combined with the others.
In order to simulate artificial biomarkers in total, all individual classes have the same number of features in total.

This is an example of simulated binary biological data including artificial biomarkers:

![Different blocks of the artificial data.](docs/source/imgs/artificial_data.png)

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

## Quickstart example

This is an example of how to generate a synthetic dataset:

```python
from biomedical_data_generator import DatasetConfig, generate_dataset

cfg = DatasetConfig(
    n_samples=30,
    n_features=200,
    n_informative=5,
    random_state=42,
)

X, y, meta = generate_dataset(cfg)
print(X.shape, y.shape, meta.n_classes)
```

## Licensing

Copyright (c) 2022 Sigrun May, Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)<br/>
Copyright (c) 2022 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften

Licensed under the **MIT License** (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License by reviewing the file
[LICENSE](https://github.com/sigrun-may/biomedical-data-generator/blob/main/LICENSE) in the repository.
