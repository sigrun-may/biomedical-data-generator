Biomedical Data Generator
=========================

Generate reproducible, labeled synthetic datasets for machine learning with a focus on biomedical applications.

.. image:: https://badge.fury.io/py/biomedical-data-generator.svg
   :target: https://pypi.org/project/biomedical-data-generator/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/biomedical-data-generator.svg
   :target: https://pypi.org/project/biomedical-data-generator/
   :alt: Python versions

.. image:: https://github.com/sigrun-may/biomedical-data-generator/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/sigrun-may/biomedical-data-generator/actions
   :alt: Tests

.. image:: https://codecov.io/gh/sigrun-may/biomedical-data-generator/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/sigrun-may/biomedical-data-generator
   :alt: Coverage

Key Features
------------

* **Correlated feature clusters** with equicorrelated and Toeplitz structures
* **Class-specific correlation patterns** (e.g., pathways active only in diseased class)
* **Batch effects simulation** with controllable confounding
* **Ground-truth metadata** capturing complete generative process
* **Scikit-learn compatible** output for seamless integration
* **Configurable feature roles** (informative, noise, proxy)

Quick Example
-------------

.. code-block:: python

   from biomedical_data_generator import DatasetConfig, ClassConfig, generate_dataset

   cfg = DatasetConfig(
       n_informative=5,
       n_noise=10,
       class_configs=[
           ClassConfig(n_samples=50, label="healthy"),
           ClassConfig(n_samples=50, label="diseased"),
       ],
       class_sep=1.5,
       random_state=42,
   )

   X, y, meta = generate_dataset(cfg)

Installation
------------

.. code-block:: bash

   pip install biomedical-data-generator

**Requirements:** Python 3.11+

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

* Generating reproducible datasets
* Simulating high-dimensional data with known ground truth feature roles
* Simulating multi-class problems with class-specific correlation structures
* Creating datasets with controlled signal-to-noise ratios

* Benchmarking feature selection and classification methods
* Evaluating methods for handling small sample sizes in high dimensions
* Testing robustness under correlation and non-causal variation
* Validating feature importance metrics against known ground truth
* Studying stability of selected features across resamples
* Exploring effects of batch confounding on model performance
* Illustrating impact of correlated features on model interpretability

* Prototyping new algorithms for biomedical data
* Generating data for domain adaptation experiments with batch effects

* Teaching machine learning concepts with transparent ground truth
* Demonstrating cross-validation pitfalls in high-dimensional settings

Scientific Context
------------------

Many biomedical machine learning problems operate in **p >> n** settings: thousands of variables but only dozens of samples. In these regimes, model performance and feature selection stability are heavily influenced by:

* Correlated feature clusters (e.g., pathways or co-expressed genes)
* Non-causal variation (batch effects, site differences)
* Noise features appearing discriminative by chance
* Small changes in class balance or effect size

This generator provides a configurable, transparent way to simulate such scenarios with complete ground truth for validation.

Architecture
------------

The generator follows a clean 6-phase pipeline:

1. **Label generation**: Create class labels with exact counts
2. **Informative features**: Generate features with class separation
3. **Correlated clusters**: Create feature blocks with within-cluster correlations
4. **Noise features**: Generate independent uninformative features
5. **Assembly**: Concatenate all feature blocks in defined order
6. **Batch effects** (optional): Apply technical overlays

Each module has single responsibility:

* ``features/informative.py``: Labels and class separation
* ``features/correlated.py``: Cluster generation with class-specific correlations
* ``features/noise.py``: Pure noise generation
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
     version      = {1.0.0}
   }

Indices
-------

* :ref:`genindex`
* :ref:`modindex`