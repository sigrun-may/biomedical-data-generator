# Biomedical Data Generator

[![PyPI version](https://badge.fury.io/py/biomedical-data-generator.svg)](https://badge.fury.io/py/biomedical-data-generator)
[![Python Version](https://img.shields.io/pypi/pyversions/biomedical-data-generator.svg)](https://pypi.org/project/biomedical-data-generator/)
[![Documentation](https://img.shields.io/badge/docs-sphinx-blue.svg)](https://sigrun-may.github.io/biomedical-data-generator/)
[![Tests](https://github.com/sigrun-may/biomedical-data-generator/actions/workflows/tests.yml/badge.svg)](https://github.com/sigrun-may/biomedical-data-generator/actions)
[![codecov](https://codecov.io/gh/sigrun-may/biomedical-data-generator/branch/main/graph/badge.svg)](https://codecov.io/gh/sigrun-may/biomedical-data-generator)

Generate reproducible synthetic biomedical datasets with known ground truth for teaching, benchmarking, and method development in high-dimensional machine learning settings.

---

## Why This Package?

Biomedical machine learning operates in challenging **p >> n** regimes (thousands of features, dozens of samples). This generator creates synthetic datasets that mimic real-world complexity while providing complete ground truth:

- **Teaching**: Demonstrate cross-validation pitfalls, feature selection stability, and batch effect impacts  
- **Benchmarking**: Compare feature selection methods with known informative features  
- **Research**: Develop and validate new algorithms with controlled data properties  
- **Reproducibility**: Deterministic generation for consistent educational materials

Compared to generic ML generators such as `sklearn.datasets.make_classification`, this package adds biomedical-specific structure: **class-specific correlations**, **explicit batch effects**, and **rich metadata** that records the full generative process (informative features, noise, correlated clusters, batch labels, configuration).

---

## Key Features

✅ **Class-specific correlations** – Simulate pathway activation only in disease states  
✅ **Batch effects** – Model technical variation with controllable confounding  
✅ **Correlated feature clusters** – Equicorrelated and Toeplitz structures  
✅ **Flexible class balance** – Exact sample counts per class  
✅ **Ground-truth metadata** – Complete generative process documentation  
✅ **scikit-learn compatible** – Seamless integration with ML pipelines

---

## Installation

```bash
pip install biomedical-data-generator
```

**Minimum Requirements:** Python 3.11+

---

## Quick Start

### Basic Dataset

```python
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
print(f"Dataset shape: {X.shape}")
print(f"True informative features: {len(meta.informative_idx)}")
```

Here, `y` contains integer-encoded class labels (`0, 1, ...`).  
If you provide human-readable labels via `ClassConfig(label=...)`, these are stored in the metadata for later interpretation.

### Class-Specific Correlations

Simulate biomarkers that only correlate in diseased patients:

```python
from biomedical_data_generator import DatasetConfig, ClassConfig, CorrClusterConfig, generate_dataset

cfg = DatasetConfig(
    n_informative=3,
    n_noise=5,
    class_configs=[
        ClassConfig(n_samples=100, label="healthy"),
        ClassConfig(n_samples=100, label="diseased"),
    ],
    corr_clusters=[
        CorrClusterConfig(
            n_cluster_features=6,
            correlation=0.2,            # baseline correlation
            class_correlation={1: 0.9}, # strong correlation in diseased class
            structure="equicorrelated",
            anchor_role="informative",
            anchor_effect_size="medium",
        )
    ],
    random_state=42,
)

X, y, meta = generate_dataset(cfg)
```

### Batch Effects

Model recruitment bias and technical variation:

```python
from biomedical_data_generator import DatasetConfig, ClassConfig, BatchEffectsConfig, generate_dataset

cfg = DatasetConfig(
    n_informative=5,
    n_noise=10,
    class_configs=[
        ClassConfig(n_samples=100, label="control"),
        ClassConfig(n_samples=100, label="disease"),
    ],
    batch_effects=BatchEffectsConfig(
        n_batches=3,
        effect_type="additive",
        effect_strength=0.5,
        confounding_with_class=0.7,  # recruitment bias
    ),
    random_state=42,
)

X, y, meta = generate_dataset(cfg)
print(f"Batch labels: {meta.batch_labels}")
```

---

## Documentation

**📖 Full documentation:** <https://sigrun-may.github.io/biomedical-data-generator/>

- [Quickstart Guide](https://sigrun-may.github.io/biomedical-data-generator/quickstart.html)  
- [API Reference](https://sigrun-may.github.io/biomedical-data-generator/api.html)  
- [Code Documentation](https://sigrun-may.github.io/biomedical-data-generator/code-doc.html)

---

## Use Cases

### Educational Applications

Ideal for teaching machine learning in biomedical contexts:

- Feature selection stability across resampling splits  
- Cross-validation pitfalls in p >> n settings  
- Batch effect impacts on model generalization  
- Correlated features and interpretability challenges  

The package is complemented by Jupyter-based teaching materials (OER) that guide learners through dataset generation, visualization, and evaluation.

### Research & Benchmarking

Systematic method comparison with known ground truth:

- Feature selection algorithm evaluation  
- Model performance under varying signal-to-noise ratios  
- Robustness testing with correlated features  
- Batch correction method validation  

---

## Scientific Context

Biomedical datasets present unique challenges:

- **High dimensionality**: p >> n creates overfitting risks  
- **Correlated features**: Biological pathways create feature clusters  
- **Batch effects**: Multi-site and multi-batch studies introduce technical variation  
- **Class imbalance**: Disease prevalence varies widely  

This generator provides realistic synthetic data that captures these properties while maintaining complete ground truth for validation. This is particularly useful when real datasets are too small, protected, or lack clear ground truth about causal vs. non-causal structure.

---

## Architecture

The generator is implemented as a six-phase pipeline with single-responsibility modules:

1. **Label generation** → Exact class counts (`DatasetConfig.class_configs`)  
2. **Informative features** → Class-separated signals  
3. **Correlated clusters** → Pathway-like structures with configurable correlation patterns  
4. **Noise features** → Independent distractors  
5. **Assembly** → Concatenation of all feature blocks into a single matrix  
6. **Batch effects (optional)** → Additive or multiplicative technical overlays, optionally confounded with class

Internally, the code is organized into dedicated modules for configuration, feature generation (informative, correlated, noise), batch effects, and metadata. A single random number generator drives the complete pipeline to ensure reproducibility.

The returned `DatasetMeta` object provides:

- Indices of informative features (e.g. `meta.informative_idx`)  
- Indices of pure-noise features  
- Indices or groupings of correlated feature clusters  
- Class and batch labels  
- A structured record of the configuration and random seeds used  

This enables precise validation of feature selection and model behavior.

---

## Examples

The `examples/` directory contains complete demonstrations:

- **01_basic_usage.py** – Simple dataset generation  
- **02_batch_effects.py** – Technical variation simulation  
- **03_class_specific_correlations.py** – Disease-specific pathway activation  
- **04_feature_selection_stability.py** – Benchmarking feature selection methods  

Run any example:

```bash
python examples/01_basic_usage.py
```

---

## Command-Line Interface

Generate datasets from YAML configuration:

```bash
bdg --config my_config.yaml --out dataset.csv
```

Example `my_config.yaml`:

```yaml
n_informative: 5
n_noise: 10
class_configs:
  - n_samples: 50
    label: "control"
  - n_samples: 50
    label: "disease"
class_sep: 1.5
random_state: 42
```

Run `bdg --help` to see all available options.

---

## Testing & Quality

The project includes a pytest-based test suite that covers:

- Informative feature generation  
- Correlated feature clusters and target correlation structures  
- Batch effect configurations and label generation  
- The scikit-learn compatible interface  

Tests are designed to ensure numerical stability, reproducibility, and consistency of the public API across releases.

---

## Citation

If you use this package in scientific work, please cite:

```bibtex
@software{biomedical_data_generator,
  author       = {May, Sigrun},
  title        = {biomedical-data-generator: Synthetic biomedical data
                  generator for benchmarking and teaching},
  year         = {2025},
  url          = {https://github.com/sigrun-may/biomedical-data-generator},
  version      = {1.0.0}
}
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/amazing-feature`)  
3. Add tests for new functionality  
4. Ensure all tests pass (`pytest`)  
5. Submit a pull request  

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## License

This project is licensed under the MIT License – see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Developed at TU Braunschweig, TU Clausthal, and Ostfalia University with support from BMBF and the State of Lower Saxony.

The project fills gaps in existing synthetic data generators by providing:

- A unified framework for class-specific correlations  
- Integrated batch effect simulation  
- An educational focus with extensive documentation  
- Complete ground truth metadata for validation  

---

## Links

- **Documentation**: <https://sigrun-may.github.io/biomedical-data-generator/>  
- **PyPI Package**: <https://pypi.org/project/biomedical-data-generator/>  
- **Issue Tracker**: <https://github.com/sigrun-may/biomedical-data-generator/issues>
