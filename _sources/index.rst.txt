Biomedical Data Generator
=========================

Generate reproducible, labeled synthetic datasets for machine learning with a focus on biomedical applications.

.. image:: https://badge.fury.io/py/biomedical-data-generator.svg
   :target: https://pypi.org/project/biomedical-data-generator/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/biomedical-data-generator.svg
   :target: https://pypi.org/project/biomedical-data-generator/
   :alt: Python versions

.. image:: https://github.com/sigrun-may/biomedical-data-generator/actions/workflows/test.yml/badge.svg
   :target: https://github.com/sigrun-may/biomedical-data-generator/actions
   :alt: Tests

.. image:: https://codecov.io/gh/sigrun-may/biomedical-data-generator/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/sigrun-may/biomedical-data-generator
   :alt: Coverage

Key Features
------------

* **Role-aware ground truth**: every column is traceable to the mechanism that
  generated it (``FeatureRoles`` and ``FeatureStrengths`` derived from metadata)
* **Channel-based signal**: informativeness is expressed structurally and *derived*,
  never declared — a ``MeanChannel`` (first moment) or ``CovarianceChannel`` (second
  moment / differential co-expression)
* **Correlated feature clusters** with equicorrelated and Toeplitz structures, plus
  attenuated anchor-to-proxy propagation
* **Signal-strength gradients** via lists of ``StandaloneInformativeGroup``
* **Class–batch confounding**: batch effects with a controllable degree of
  correlation between batch assignment and class label
* **Scikit-learn compatible** output for seamless integration

Quick Example
-------------

.. code-block:: python

   from biomedical_data_generator import (
       DatasetConfig,
       ClassConfig,
       CorrClusterConfig,
       MeanChannel,
       StandaloneInformativeGroup,
       generate_dataset,
       compute_feature_roles,
   )

   cfg = DatasetConfig(
       # A signal-strength gradient: strong, medium, weak groups.
       standalone_informative_groups=[
           StandaloneInformativeGroup(n_features=3, class_sep=2.0),
           StandaloneInformativeGroup(n_features=3, class_sep=1.0),
           StandaloneInformativeGroup(n_features=3, class_sep=0.4),
       ],
       n_standalone_noise=10,
       class_configs=[
           ClassConfig(n_samples=50, label="healthy"),
           ClassConfig(n_samples=50, label="diseased"),
       ],
       corr_clusters=[
           # Made informative through a mean shift on the diseased class.
           CorrClusterConfig(
               n_cluster_features=4,
               baseline_correlation=0.6,
               mean_channel=MeanChannel(per_class_effect={1: 1.5}),
               label="Pathway_A",
           ),
       ],
       random_state=42,
   )

   X, y, meta = generate_dataset(cfg)
   roles = compute_feature_roles(meta)  # derived six-way column partition

Installation
------------

.. code-block:: bash

   pip install biomedical-data-generator

**Requirements:** Python 3.10+

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   code-doc

.. toctree::
   :maxdepth: 1
   :caption: External Links

   GitHub Repository <https://github.com/sigrun-may/biomedical-data-generator>
   PyPI Package <https://pypi.org/project/biomedical-data-generator/>
   License <https://github.com/sigrun-may/biomedical-data-generator/blob/main/LICENSE>

Use Cases
---------

This package is designed for:

* Validating feature-importance and feature-selection methods against known,
  role-aware ground truth (which columns carry signal, and through which channel)
* Separating first-moment (mean) from second-moment (differential co-expression)
  signal when evaluating detectors
* Simulating multi-class problems with class-specific correlation structures
* Creating datasets with controlled, derived signal-to-noise structure

* Benchmarking batch-correction methods under controllable **class–batch
  confounding**, where non-causal variation correlates with the label
* Exposing models that latch onto batch or correlated proxies rather than the
  causal anchor
* Studying stability of selected features across resamples
* Illustrating the impact of correlated proxies on model interpretability

* Prototyping new algorithms for biomedical data
* Generating data for domain-adaptation experiments with batch effects
* Teaching machine learning concepts with transparent, traceable ground truth

Scientific Context
------------------

Biomedical machine learning typically operates in **p ≫ n** settings: many
variables (genes, proteins, metabolites) measured on comparatively few samples.
In these settings, model behavior and feature-selection stability are shaped by:

* Correlated feature clusters (e.g., pathways or co-expressed genes)
* Non-causal variation (batch effects, site differences) that may confound with class
* First- vs. second-moment signal (mean shifts vs. differential co-expression)
* Correlated proxies that mimic a causal anchor

What sets this generator apart is **role-aware ground truth** — every column is
traceable to the mechanism that generated it — and **explicit class–batch
confounding**, so non-causal variation can be dialed in and measured against the
truth rather than inferred.

Architecture
------------

The generator pipeline:

1. **Informative features + labels**: Generate class-separated informative features and class labels (exact per-class counts)
2. **Correlated clusters**: Create feature blocks with within-cluster correlations
3. **Noise features**: Generate independent uninformative features
4. **Assembly**: Concatenate all feature blocks in defined order
5. **Batch effects** (optional): Apply technical overlays

Each module has single responsibility:

* ``features/informative.py``: Labels and class separation
* ``features/correlated.py``: Cluster generation with class-specific correlations
* ``utils/sampling.py``: Distribution sampling (used for noise features)
* ``effects/batch.py``: Technical overlays (batch effects)
* ``generator.py``: Pipeline orchestration
* ``config.py``: Configuration models with validation
* ``meta.py``: Ground truth capture

Citation
--------

If you use this package in a scientific publication, please cite:

.. code-block:: bibtex

   @software{biomedical_data_generator,
     author       = {May, Sigrun},
     title        = {biomedical-data-generator: Synthetic biomedical data
                     generator for benchmarking and teaching},
     year         = {2025},
     url          = {https://github.com/sigrun-may/biomedical-data-generator},
     version      = {2.0.0}
   }

Indices
-------

* :ref:`genindex`
* :ref:`modindex`