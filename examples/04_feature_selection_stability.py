# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Feature selection stability example for biomedical-data-generator.

This example demonstrates:
- Testing feature selection algorithms with known ground truth
- Evaluating stability of feature selection across different random splits
- Comparing performance with varying signal-to-noise ratios
- Assessing how correlation structure affects feature selection
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.feature_selection import SelectKBest, f_classif  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedKFold  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from biomedical_data_generator.config import ClassConfig, CorrClusterConfig, DatasetConfig
from biomedical_data_generator.generator import generate_dataset


def evaluate_feature_selection_stability(X, y, meta, n_features: int = 10, n_splits: int = 10) -> dict[str, dict]:
    """Evaluate stability of feature selection across multiple CV splits.

    Args:
        X: Feature matrix
        y: Target labels
        meta: Dataset metadata with ground truth feature roles
        n_features: Number of features to select
        n_splits: Number of cross-validation splits

    Returns:
        Dictionary with stability metrics for each method
    """
    methods_names = {
        "ANOVA F-test": SelectKBest(f_classif, k=n_features),
        "Random Forest": None,  # Will be handled separately
    }

    results_raw: dict[str, dict[str, list[Any]]] = {
        method_name: {"selected_features": [], "precision": [], "recall": []} for method_name in methods_names
    }

    # True informative features (ground truth)
    true_informative: set[int] = set(meta.informative_idx)

    # Cross-validation splits
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # ANOVA F-test
        selector = SelectKBest(f_classif, k=n_features)
        selector.fit(X_train_scaled, y_train)
        selected = set(selector.get_support(indices=True))
        results_raw["ANOVA F-test"]["selected_features"].append(selected)

        # Calculate precision and recall
        true_positives = len(selected & true_informative)
        precision = true_positives / len(selected) if len(selected) > 0 else 0
        recall = true_positives / len(true_informative) if len(true_informative) > 0 else 0

        results_raw["ANOVA F-test"]["precision"].append(precision)
        results_raw["ANOVA F-test"]["recall"].append(recall)

        # Random Forest feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42 + fold_idx, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        importances = rf.feature_importances_
        top_k_indices = np.argsort(importances)[-n_features:]
        selected_rf = set(top_k_indices)
        results_raw["Random Forest"]["selected_features"].append(selected_rf)

        # Calculate precision and recall for RF
        true_positives_rf = len(selected_rf & true_informative)
        precision_rf = true_positives_rf / len(selected_rf) if len(selected_rf) > 0 else 0
        recall_rf = true_positives_rf / len(true_informative) if len(true_informative) > 0 else 0

        results_raw["Random Forest"]["precision"].append(precision_rf)
        results_raw["Random Forest"]["recall"].append(recall_rf)

    # Calculate stability (Jaccard similarity between splits)
    final_results: dict[str, dict[str, float]] = {}
    for method_name, method_results in results_raw.items():
        selected_sets = method_results["selected_features"]
        jaccard_scores = []

        for i in range(len(selected_sets)):
            for j in range(i + 1, len(selected_sets)):
                intersection = len(selected_sets[i] & selected_sets[j])
                union = len(selected_sets[i] | selected_sets[j])
                jaccard = intersection / union if union > 0 else 0
                jaccard_scores.append(jaccard)

        final_results[method_name] = {
            "stability": float(np.mean(jaccard_scores) if jaccard_scores else 0.0),
            "mean_precision": float(np.mean(method_results["precision"])) if method_results["precision"] else 0.0,
            "mean_recall": float(np.mean(method_results["recall"])) if method_results["recall"] else 0.0,
        }
    return final_results


def main() -> None:
    """Demonstrate feature selection stability testing with synthetic data."""
    print("=" * 70)
    print("Feature Selection Stability: Testing with Known Ground Truth")
    print("=" * 70)
    print()

    # Scenario 1: High signal-to-noise ratio (easy case)
    print("Scenario 1: High Signal-to-Noise Ratio")
    print("-" * 70)

    cfg_high_snr = DatasetConfig(
        n_informative=10,
        n_noise=40,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=5,
                correlation=0.7,
                structure="equicorrelated",
                anchor_role="informative",
                anchor_effect_size="large",
                label="Strong Signal Pathway",
            )
        ],
        class_configs=[
            ClassConfig(n_samples=150, label="control"),
            ClassConfig(n_samples=150, label="disease"),
        ],
        class_sep=[2.0],  # Large separation
        random_state=42,
    )

    X_high, y_high, meta_high = generate_dataset(cfg_high_snr)
    print(f"✓ Generated dataset: {X_high.shape}")
    print(f"  - True informative features: {len(meta_high.informative_idx)}")
    print(f"  - Noise features: {len(meta_high.noise_idx)}")
    print()

    print("Evaluating feature selection stability...")
    results_high = evaluate_feature_selection_stability(X_high, y_high, meta_high, n_features=10, n_splits=10)

    print("\nResults for High SNR:")
    for method, metrics in results_high.items():
        print(f"  {method}:")
        print(f"    - Mean Precision: {metrics['mean_precision']:.3f}")
        print(f"    - Mean Recall:    {metrics['mean_recall']:.3f}")
        print(f"    - Stability:      {metrics['stability']:.3f}")
    print()

    # Scenario 2: Low signal-to-noise ratio (challenging case)
    print("Scenario 2: Low Signal-to-Noise Ratio")
    print("-" * 70)

    cfg_low_snr = DatasetConfig(
        n_informative=5,
        n_noise=45,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=3,
                correlation=0.5,
                structure="equicorrelated",
                anchor_role="informative",
                anchor_effect_size="small",
                label="Weak Signal Pathway",
            )
        ],
        class_configs=[
            ClassConfig(n_samples=100, label="control"),
            ClassConfig(n_samples=100, label="disease"),
        ],
        class_sep=[0.5],  # Small separation
        random_state=42,
    )

    X_low, y_low, meta_low = generate_dataset(cfg_low_snr)
    print(f"✓ Generated dataset: {X_low.shape}")
    print(f"  - True informative features: {len(meta_low.informative_idx)}")
    print(f"  - Noise features: {len(meta_low.noise_idx)}")
    print()

    print("Evaluating feature selection stability...")
    results_low = evaluate_feature_selection_stability(X_low, y_low, meta_low, n_features=10, n_splits=10)

    print("\nResults for Low SNR:")
    for method, metrics in results_low.items():
        print(f"  {method}:")
        print(f"    - Mean Precision: {metrics['mean_precision']:.3f}")
        print(f"    - Mean Recall:    {metrics['mean_recall']:.3f}")
        print(f"    - Stability:      {metrics['stability']:.3f}")
    print()

    # Scenario 3: Highly correlated features (redundancy challenge)
    print("Scenario 3: Highly Correlated Features (Redundancy)")
    print("-" * 70)

    cfg_corr = DatasetConfig(
        n_informative=8,
        n_noise=30,
        corr_clusters=[
            CorrClusterConfig(
                n_cluster_features=6,
                correlation=0.95,  # Very high correlation
                structure="equicorrelated",
                anchor_role="informative",
                anchor_effect_size="medium",
                label="Highly Redundant Pathway",
            )
        ],
        class_configs=[
            ClassConfig(n_samples=120, label="control"),
            ClassConfig(n_samples=120, label="disease"),
        ],
        class_sep=[1.5],
        random_state=42,
    )

    X_corr, y_corr, meta_corr = generate_dataset(cfg_corr)
    print(f"✓ Generated dataset: {X_corr.shape}")
    print(f"  - True informative features: {len(meta_corr.informative_idx)}")
    print(f"  - Correlated cluster size: {len(meta_corr.corr_cluster_indices[0])}")  # Cluster ID 0 (0-based)
    print()

    print("Evaluating feature selection stability...")
    results_corr = evaluate_feature_selection_stability(X_corr, y_corr, meta_corr, n_features=10, n_splits=10)

    print("\nResults for Highly Correlated Features:")
    for method, metrics in results_corr.items():
        print(f"  {method}:")
        print(f"    - Mean Precision: {metrics['mean_precision']:.3f}")
        print(f"    - Mean Recall:    {metrics['mean_recall']:.3f}")
        print(f"    - Stability:      {metrics['stability']:.3f}")
    print()

    # Visualization
    print("=" * 70)
    print("Generating Comparison Visualizations...")
    print("-" * 70)

    try:
        scenarios = ["High SNR", "Low SNR", "High Correlation"]
        results_list = [results_high, results_low, results_corr]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        metric_names = ["mean_precision", "mean_recall", "stability"]
        metric_labels = ["Precision", "Recall", "Stability"]

        for ax, metric, label in zip(axes, metric_names, metric_labels):
            x = np.arange(len(scenarios))
            width = 0.35

            method_names_list: list[str] = list(results_high.keys())
            for i, method in enumerate(method_names_list):
                values = [results[method][metric] for results in results_list]
                ax.bar(x + i * width, values, width, label=method, alpha=0.8)

            ax.set_ylabel(label)
            ax.set_title(f"{label} Across Scenarios")
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(scenarios, rotation=15, ha="right")
            ax.legend()
            ax.set_ylim([0, 1.1])
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig("feature_selection_stability_comparison.png", dpi=150)
        print("✓ Saved comparison visualization to feature_selection_stability_comparison.png")
        print()

    except Exception as e:
        print(f"Note: Visualization skipped ({e})")
        print()

    # Save datasets
    datasets = [
        (X_high, y_high, "high_snr"),
        (X_low, y_low, "low_snr"),
        (X_corr, y_corr, "high_correlation"),
    ]

    for i, (X, y, name) in enumerate(datasets, start=1):
        out_path = f"feature_selection_example_{i}_{name}.csv"
        df_out = X.copy()
        df_out["target"] = y
        df_out.to_csv(out_path, index=False)  # type: ignore
        print(f"✓ Saved {name} dataset to {out_path}")

    print()
    print("=" * 70)
    print("Key Takeaways:")
    print("  1. Feature selection stability varies with signal-to-noise ratio")
    print("  2. High correlation creates redundancy, reducing stability")
    print("  3. Ground truth allows quantitative evaluation (precision/recall)")
    print("  4. Synthetic data enables controlled testing of FS algorithms")
    print("  5. Stability (Jaccard index) measures consistency across splits")
    print("=" * 70)


if __name__ == "__main__":
    main()
