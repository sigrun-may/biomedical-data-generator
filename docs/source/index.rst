Biomedical Data Generator
=========================

A small library to generate synthetic classification datasets with correlated feature clusters.

Content
-------

.. toctree::
   :glob:
   :maxdepth: 2

   quickstart
   api

   code-doc
   License <https://github.com/sigrun-may/biomedical-data-generator/blob/main/LICENSE>
   GitHub Repository <https://github.com/sigrun-may/biomedical-data-generator>

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`


**`docs/api.rst`**
```rst
API Reference
=============

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   biomedical_data_generator.generate_dataset
   biomedical_data_generator.generate_correlated_cluster
   biomedical_data_generator.find_seed_for_correlation
   biomedical_data_generator.find_dataset_seed_for_class_weights
   biomedical_data_generator.find_dataset_seed_for_score
   biomedical_data_generator.DatasetConfig
   biomedical_data_generator.CorrCluster
   biomedical_data_generator.DatasetMeta
