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
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   biomedical_data_generator.features.noise

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
   biomedical_data_generator.utils.visualization
   biomedical_data_generator.utils.sklearn_compat


