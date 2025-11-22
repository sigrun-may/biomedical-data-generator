# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Smoke tests for example scripts.

These tests ensure that all example scripts run without errors and produce
expected outputs. They serve as integration tests and prevent examples from
becoming stale as the codebase evolves.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch

import matplotlib
import pytest

# Use non-interactive backend for testing
matplotlib.use("Agg")

# Add examples directory to path
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def import_example(example_name: str):
    """Import an example module dynamically (handles names starting with digits)."""
    example_path = EXAMPLES_DIR / f"{example_name}.py"
    spec = importlib.util.spec_from_file_location(example_name, example_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {example_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[example_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def cleanup_output_files():
    """Clean up any CSV or image files created during tests."""
    yield
    # Clean up generated files
    patterns = ["*.csv", "*.png"]
    for pattern in patterns:
        for filepath in Path.cwd().glob(pattern):
            try:
                filepath.unlink()
            except Exception:
                pass


@pytest.fixture
def suppress_matplotlib_show():
    """Prevent matplotlib from trying to display plots during tests."""
    with patch("matplotlib.pyplot.show"):
        yield


def test_01_basic_usage_runs(capsys, suppress_matplotlib_show):
    """Test that 01_basic_usage.py runs without errors."""
    # Import and run the example
    example = import_example("01_basic_usage")

    example.main()

    # Check that output was printed
    captured = capsys.readouterr()
    assert "Basic Usage Example" in captured.out
    assert "Generated dataset with shape" in captured.out
    assert "Feature Roles:" in captured.out
    assert "Class Distribution:" in captured.out

    # Check that CSV file was created
    assert Path("basic_dataset.csv").exists()


def test_02_batch_effects_runs(capsys, suppress_matplotlib_show):
    """Test that 02_batch_effects.py runs without errors."""
    example = import_example("02_batch_effects")

    example.main()

    # Check that output was printed
    captured = capsys.readouterr()
    assert "Batch Effects Example" in captured.out
    assert "Random Batch Assignment" in captured.out
    assert "Confounded Batch Assignment" in captured.out
    assert "Multiplicative Batch Effects" in captured.out

    # Check that CSV files were created
    expected_files = [
        "batch_example_1_random_batches.csv",
        "batch_example_2_confounded_batches.csv",
        "batch_example_3_multiplicative_batches.csv",
        "batch_example_4_informative_only_batches.csv",
    ]
    for filename in expected_files:
        assert Path(filename).exists(), f"Expected file {filename} not found"


def test_03_class_specific_correlations_runs(capsys, suppress_matplotlib_show):
    """Test that 03_class_specific_correlations.py runs without errors."""
    example = import_example("03_class_specific_correlations")

    example.main()

    # Check that output was printed
    captured = capsys.readouterr()
    assert "Class-Specific Correlations" in captured.out
    assert "Pathway Active Only in Diseased State" in captured.out
    assert "Different Correlation Strengths" in captured.out
    assert "Multiple Pathways" in captured.out

    # Check that CSV files were created
    expected_files = [
        "class_specific_example_1_disease_activated_pathway.csv",
        "class_specific_example_2_progressive_correlation.csv",
        "class_specific_example_3_multiple_pathways.csv",
    ]
    for filename in expected_files:
        assert Path(filename).exists(), f"Expected file {filename} not found"


def test_04_feature_selection_stability_runs(capsys, suppress_matplotlib_show):
    """Test that 04_feature_selection_stability.py runs without errors."""
    example = import_example("04_feature_selection_stability")

    example.main()

    # Check that output was printed
    captured = capsys.readouterr()
    assert "Feature Selection Stability" in captured.out
    assert "High Signal-to-Noise Ratio" in captured.out
    assert "Low Signal-to-Noise Ratio" in captured.out
    assert "Highly Correlated Features" in captured.out
    assert "Mean Precision" in captured.out
    assert "Mean Recall" in captured.out
    assert "Stability" in captured.out

    # Check that CSV files were created
    expected_files = [
        "feature_selection_example_1_high_snr.csv",
        "feature_selection_example_2_low_snr.csv",
        "feature_selection_example_3_high_correlation.csv",
    ]
    for filename in expected_files:
        assert Path(filename).exists(), f"Expected file {filename} not found"


def test_02_batch_effects_create_dataset_with_batches():
    """Test the helper function in 02_batch_effects.py."""
    example = import_example("02_batch_effects")

    X, y, meta, cfg = example.create_dataset_with_batches(
        n_batches=2, confounding=0.5, effect_type="additive", effect_strength=0.3
    )

    # Check basic properties
    assert X.shape[0] == 200  # 100 + 100 samples
    assert len(y) == 200
    assert cfg.batch is not None
    assert cfg.batch.n_batches == 2
    assert cfg.batch.effect_strength == 0.3


def test_03_compute_correlation_by_class():
    """Test the helper function in 03_class_specific_correlations.py."""
    import pandas as pd

    example = import_example("03_class_specific_correlations")

    # Create simple test data
    X = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [2, 4, 6, 8], "f3": [1, 1, 2, 2]})
    y = pd.Series(["A", "A", "B", "B"])

    correlations = example.compute_correlation_by_class(X, y, [0, 1])

    # Check that we got correlations for each class
    assert "A" in correlations
    assert "B" in correlations
    assert correlations["A"].shape == (2, 2)  # 2 features
    assert correlations["B"].shape == (2, 2)


def test_04_evaluate_feature_selection_stability():
    """Test the feature selection stability evaluation function."""
    import pandas as pd

    from biomedical_data_generator import ClassConfig, DatasetConfig, generate_dataset

    example = import_example("04_feature_selection_stability")

    # Create a simple dataset
    cfg = DatasetConfig(
        n_informative=5,
        n_noise=5,
        class_configs=[
            ClassConfig(n_samples=50, label="A"),
            ClassConfig(n_samples=50, label="B"),
        ],
        random_state=42,
    )
    X, y, meta = generate_dataset(cfg)

    # Run stability evaluation with fewer splits for speed
    results = example.evaluate_feature_selection_stability(X, y, meta, n_features=5, n_splits=3)

    # Check that results were computed
    assert "ANOVA F-test" in results
    assert "Random Forest" in results

    for method in results:
        assert "mean_precision" in results[method]
        assert "mean_recall" in results[method]
        assert "stability" in results[method]
        assert 0 <= results[method]["mean_precision"] <= 1
        assert 0 <= results[method]["mean_recall"] <= 1
        assert 0 <= results[method]["stability"] <= 1


@pytest.mark.parametrize(
    "example_name",
    [
        "01_basic_usage",
        "02_batch_effects",
        "03_class_specific_correlations",
        "04_feature_selection_stability",
    ],
)
def test_example_has_main_guard(example_name):
    """Ensure each example has proper if __name__ == '__main__' guard."""
    example_file = EXAMPLES_DIR / f"{example_name}.py"
    content = example_file.read_text()

    assert 'if __name__ == "__main__":' in content
    assert "def main()" in content


@pytest.mark.parametrize(
    "example_name",
    [
        "01_basic_usage",
        "02_batch_effects",
        "03_class_specific_correlations",
        "04_feature_selection_stability",
    ],
)
def test_example_has_copyright_and_docstring(example_name):
    """Ensure each example has copyright notice and module docstring."""
    example_file = EXAMPLES_DIR / f"{example_name}.py"
    content = example_file.read_text()

    assert "Copyright (c) 2025 Sigrun May" in content
    assert "MIT license" in content
    assert '"""' in content  # Has docstring
