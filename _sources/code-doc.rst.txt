Code Documentation
==================

This section provides a complete overview of the internal modules of
``biomedical-data-generator``.
It is intended for developers, contributors, and advanced users who want
to understand or extend the code base.

The API documentation is automatically generated using Sphinx
``autodoc`` and ``autosummary``.
Each module listed below expands into a separate page in the
``_autosummary`` directory.

---

Configuration Models
--------------------

These classes define the full dataset configuration, including
class structure, correlated clusters, noise distribution, and optional
batch effects.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   biomedical_data_generator.config.ClassConfig
   biomedical_data_generator.config.BatchEffectsConfig
   biomedical_data_generator.config.CorrClusterConfig
   biomedical_data_generator.config.DatasetConfig

Cluster signal channels
~~~~~~~~~~~~~~~~~~~~~~~~~

A correlated cluster carries class-discriminating signal only through its
optional channels. The **mean channel** encodes a first-moment (per-class
mean shift on the anchor); the **covariance channel** encodes a second-moment
(per-class within-cluster correlation, i.e. differential co-expression).
Both are resolved per class by :class:`~biomedical_data_generator.config.CorrClusterConfig`,
falling back to a baseline when a class is absent.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   biomedical_data_generator.config.MeanChannel
   biomedical_data_generator.config.CovarianceChannel

Standalone informative groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Groups of cluster-free informative features that share one separation
strength. A list of groups with decreasing ``class_sep`` realizes a
signal-strength gradient across the standalone-informative block.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   biomedical_data_generator.config.StandaloneInformativeGroup

---

Dataset Generator
-----------------

The central entry point for creating synthetic datasets.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   biomedical_data_generator.generate_dataset

---

Feature Generators
------------------

Functions responsible for generating informative features,
noise features, and correlated feature clusters.

Informative features
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   biomedical_data_generator.features.informative

Correlated feature clusters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   biomedical_data_generator.features.correlated

Independent noise features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Independent noise features are not produced by a dedicated module. They are
sampled directly in :func:`biomedical_data_generator.generate_dataset` using
:func:`biomedical_data_generator.utils.sampling.sample_distribution`.

---

Batch Effects
-------------

Simulation of site effects, instrument variation, temporal drift,
and confounding with class labels.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   biomedical_data_generator.effects.batch

---

Metadata
--------

Structured metadata describing the full generative process, including
feature roles, class labels, correlated clusters, batch labels,
and derived dataset properties.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   biomedical_data_generator.meta.DatasetMeta
   biomedical_data_generator.meta.FeatureRoles
   biomedical_data_generator.meta.compute_feature_roles

Per-feature signal strengths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Derived, per-column assessment of how strongly each feature separates the
classes. :func:`~biomedical_data_generator.meta.compute_feature_strengths`
reads a :class:`~biomedical_data_generator.meta.DatasetMeta` record and
returns a :class:`~biomedical_data_generator.meta.FeatureStrengths` summary.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   biomedical_data_generator.meta.FeatureStrengths
   biomedical_data_generator.meta.compute_feature_strengths

---

Utility Modules (Optional)
--------------------------
Helper functions for data manipulation, visualization, and
integration with scikit-learn.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   biomedical_data_generator.utils.correlation_tools
   biomedical_data_generator.utils.export_utils
   biomedical_data_generator.utils.sampling
   biomedical_data_generator.utils.visualization
   biomedical_data_generator.utils.sklearn_compat


