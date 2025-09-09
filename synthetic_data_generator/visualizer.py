# Copyright (c) 2022 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Visualizer for the generated artificial data.

Visualizes correlation matrices of the correlated feature clusters and class histograms.
"""
import pandas as pd
import seaborn as sns
from matplotlib import pyplot


def visualize(data_df, params_dict, path=None) -> None:
    """Visualize generated artificial biomarker data.

    Args:
        data_df: DataFrame where each column equals a feature or biomarker candidate. The features must not be shuffled.
        params_dict: The input parameters for the generated artificial data.
        path: Path to folder to save the figures.

    Returns:
        None
    """
    # shuffled features cannot be visualized semantically correct
    assert not params_dict["shuffle_features"]
    classes_df = pd.DataFrame()
    start_row = 0
    start_column = 1  # first column (zero) contains the label
    stop_row = params_dict["classes"][1]["number_of_samples"]

    # First column (zero) contains the label. Therefore, the stop column is shifted by 1.
    stop_column = 1

    for class_number in params_dict["classes"]:
        if params_dict["classes"][class_number]["correlated_features"]:
            for cluster_number in params_dict["classes"][class_number]["correlated_features"]:
                stop_column += params_dict["classes"][class_number]["correlated_features"][cluster_number][
                    "number_of_features"
                ]
                sub_df = data_df.iloc[start_row:stop_row, start_column:stop_column]
                if path:
                    complete_path = f"{path}/corrplot_class{class_number}_cluster{cluster_number}.pdf"
                    visualize_correlation_matrix(sub_df, complete_path, annotate=True)
                else:
                    visualize_correlation_matrix(sub_df, annotate=True)
                # shift column start index
                start_column += params_dict["classes"][class_number]["correlated_features"][cluster_number][
                    "number_of_features"
                ]

            class_df = data_df.iloc[
                start_row:stop_row, 1 : params_dict["number_of_relevant_features"] + 1
            ]  # skip the label
            if path:
                complete_path = f"{path}/corrplot_class{class_number}.pdf"
                visualize_correlation_matrix(class_df, complete_path, annotate=False)
            else:
                visualize_correlation_matrix(class_df, annotate=False)
        class_values = (
            data_df.iloc[start_row:stop_row, 1 : params_dict["number_of_relevant_features"] + 1].to_numpy().flatten()
        )
        column_name = f"class_{class_number} mode {params_dict['classes'][class_number]['mode']}"
        classes_df[column_name] = class_values

        # shift start and stop indices
        start_row += params_dict["classes"][class_number]["number_of_samples"]
        # skip stop_row for last class
        if class_number + 1 in params_dict["classes"].keys():
            stop_row += params_dict["classes"][class_number + 1]["number_of_samples"]

    sns.histplot(data=classes_df)
    if path:
        complete_path = f"{path}/classes_histogram.pdf"
        pyplot.savefig(complete_path, dpi=600)
    pyplot.show()


def visualize_correlation_matrix(data_df: pd.DataFrame, path: str = None, annotate: bool = False) -> None:
    """Visualize correlations.

    Args:
        data_df: DataFrame where each column equals a class.
        path: Path to save the figure
        annotate: If the correlation matrix should be annotated.

    Returns:
        None
    """
    sns.set_theme(style="white")
    corr = data_df.corr(method="spearman")
    sns.heatmap(corr, annot=annotate, cmap="Blues", fmt=".2g")
    if path:
        pyplot.savefig(path, dpi=600)
    pyplot.show()
