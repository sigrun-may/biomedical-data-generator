# Copyright (c) 2023 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""This module provides tools for generating artificial data for classification tasks.

It includes functions for generating correlated features, normal distributed features,
lognormal distributed features, and random noise features.
It also provides functions for visualizing the generated data,
including plotting the distribution of class features and the correlation between classes.
The main function in this module is generate_artificial_classification_data,
which generates a complete dataset with a specified number of classes,
each with a specified number of samples and features.
The generated data can be used for benchmarking and development of new methods
in machine learning and data analysis.
"""
from __future__ import annotations

import logging
import math
import warnings
from typing import Literal, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from numpy import ndarray
from statsmodels.stats import correlation_tools


def generate_correlated_cluster(
    number_of_features: int,
    number_of_samples: int,
    lower_bound: float,
    upper_bound: float,
    plot=True,
    show_values=True,
    path_to_save_pdf="",
) -> ndarray:
    """Generate a cluster of correlated features.

    Args:
        number_of_features: Number of columns of generated data.
        number_of_samples: Number of rows of generated data.
        lower_bound: Lower bound of the generated correlations.
        upper_bound: Upper bound of the generated correlations.
        plot: Plot the generated cluster of correlated features.
        show_values: Show the correlation values in the visualization.
        path_to_save_pdf: Path to save the visualization as pdf.

    Returns:
        Numpy array of the given shape with correlating features in the given range.
    """
    rng = np.random.default_rng()

    # generate random matrix to constrain a range of values
    # and specify a starting point
    random_matrix = rng.uniform(
        low=lower_bound,
        high=upper_bound,
        size=(number_of_features, number_of_features),
    )
    print("generation of correlation matrix ...")
    # first iteration generating correlations to
    # improve the fit of the covariance matrix
    correlation_matrix = correlation_tools.corr_nearest(corr=random_matrix, threshold=1e-15, n_fact=10000)

    # change values on the diagonal to 1 to
    # improve the fit of the covariance matrix
    for i in range(0, correlation_matrix.shape[0]):
        correlation_matrix[i, i] = 1

    print("generation of covariant matrix ...")
    # generate the nearest positive semi-definite covariance matrix
    covariance_matrix = correlation_tools.cov_nearest(
        correlation_matrix,
        method="nearest",
        threshold=1e-15,
        n_fact=10000,
        return_all=False,
    )
    # generate correlated cluster
    covariant_cluster = rng.multivariate_normal(
        mean=np.zeros(number_of_features),
        cov=covariance_matrix,
        size=number_of_samples,
        check_valid="raise",
        method="eigh",
    )
    assert covariant_cluster.shape[0] == number_of_samples, (
        f"Number of rows of generated covariant cluster {covariant_cluster.shape[0]} "
        f"does not match number of samples {number_of_samples}"
    )
    assert covariant_cluster.shape[1] == number_of_features, (
        f"Number of columns of generated covariant cluster {covariant_cluster.shape[1]} "
        f"does not match number of features {number_of_features}"
    )
    if plot:
        plot_correlated_cluster(covariant_cluster, show_values=show_values, path_to_save_pdf=path_to_save_pdf)
    return covariant_cluster


def generate_normal_distributed_informative_features_for_one_class(
    number_of_samples: int, number_of_normal_distributed_relevant_features: int, scale: float = 1.0, plot=True
) -> ndarray:
    """Generate normal distributed informative features for one class with the given scale.

    Args:
        number_of_samples: Number of rows of generated data.
        number_of_normal_distributed_relevant_features: Number of columns of generated data.
        scale: Scale of the normal distribution.
        plot: Plot the generated normal distributed informative features.

    Returns:
        Numpy array of the given shape with normal distributed class features.
    """
    # check if number of relevant features is greater than zero
    if not number_of_normal_distributed_relevant_features > 0:
        raise ValueError("Number of relevant features must be greater than zero.")

    if not number_of_samples > 0:
        raise ValueError("Number of samples must be greater than zero.")

    if not scale > 0:
        raise ValueError("Scale must be greater than zero.")

    # generate normal distributed random data
    rng = np.random.default_rng()
    relevant_features_np = rng.normal(
        loc=0,
        scale=scale,
        size=(number_of_samples, number_of_normal_distributed_relevant_features),
    )

    if not math.isclose(np.mean(relevant_features_np), 0, abs_tol=0.15):
        warnings.warn(
            f"mean of generated data {str(np.mean(relevant_features_np))} differs from expected mean {str(0)} "
            f"-> Try choosing a smaller scale for a small sample size or accept a deviating mean. "
            f"The current scale is {str(scale)}."
        )

    if plot:
        plot_distribution_of_class_features_for_single_class(relevant_features_np)

    return relevant_features_np


def transform_normal_distributed_class_features_to_lognormal_distribution(
    class_features_data_array: ndarray,
) -> ndarray:
    """Transform the given normal distributed class features to a lognormal distribution to simulate outliers.

    Args:
        class_features_data_array: Normal distributed class features to transform.

    Returns:
        Numpy array of the given shape with lognormal distributed class features.
    """
    lognormal_distributed_class_features_np = np.exp(class_features_data_array)
    return lognormal_distributed_class_features_np


def shift_class_to_enlarge_effectsize(class_features_np: ndarray, effect_size: float) -> ndarray:
    """Shift the given class features to simulate the given effect size.

    Args:
        class_features_np: Class features to shift.
        effect_size: Effect size to shift the class features to.

    Returns:
        Numpy array of the given shape with shifted class features.
    """
    class_features_np = class_features_np + effect_size
    return class_features_np


def build_class(
    number_of_samples_per_class: int,
    number_of_informative_features: int,
    scale: float = 1,
    distribution: Literal["normal", "lognormal"] = "normal",
    correlated_clusters_list: list|None = None,
    plot_correlation_matrix: bool = True,
    plot_distribution: bool = True,
    path_to_save_pdf="",
) -> ndarray:
    """Build a class with the given number of samples per class and the given number of informative features.

    Args:
        number_of_samples_per_class: Number of samples per class.
        number_of_informative_features: Number of informative features.
        scale: Scale of the normal distribution.
        distribution: Distribution of the class features. Possible values are "normal" and "lognormal".
        correlated_clusters_list: List of correlated clusters to include in the class.
        plot_correlation_matrix: Plot the correlation matrix of the correlated class features.
        plot_distribution: Plot the distribution of the class features.
        path_to_save_pdf: Path to save the visualization as pdf.

    Returns:
        Numpy array of the given shape with generated features for the class.
    """
    # check if number of relevant features is greater than zero
    if not number_of_informative_features > 0:
        raise ValueError("Number of informative features must be greater than zero.")

    # check if number of samples per class is greater than zero
    if not number_of_samples_per_class > 0:
        raise ValueError("Number of samples per class must be greater than zero.")

    # check if scale is greater than zero
    if not scale > 0:
        raise ValueError("Scale must be greater than zero.")

    # check if a plot will be generated if a plot path is given
    if path_to_save_pdf != "" and not (plot_correlation_matrix or plot_distribution):
        raise ValueError(
            "No plot will be generated. " "Please set 'plot_correlation_matrix' or 'plot_distribution' to True."
        )

    # check if path to save pdf is a string
    if not isinstance(path_to_save_pdf, str):
        raise ValueError("Path to save pdf must be a string.")

    # check if path to save pdf is valid
    if path_to_save_pdf != "":
        if not path_to_save_pdf.endswith(".pdf"):
            raise ValueError("Path to save pdf must end with '.pdf'.")

    class_features_list = []
    if correlated_clusters_list is not None:
        correlated_clusters = np.concatenate(correlated_clusters_list, axis=1)
        if plot_correlation_matrix:
            print("Correlation matrix of correlated clusters:")
            plot_correlated_cluster(correlated_clusters, show_values=False, path_to_save_pdf=path_to_save_pdf)
        class_features_list.append(correlated_clusters)
        number_of_normal_distributed_relevant_features = number_of_informative_features - correlated_clusters.shape[1]
    else:
        number_of_normal_distributed_relevant_features = number_of_informative_features

    if number_of_normal_distributed_relevant_features > 0:
        class_data_array = generate_normal_distributed_informative_features_for_one_class(
            number_of_samples=number_of_samples_per_class,
            number_of_normal_distributed_relevant_features=number_of_normal_distributed_relevant_features,
            scale=scale,
            plot=False,
        )
        class_features_list.append(class_data_array)

    class_features = np.concatenate(class_features_list, axis=1)
    assert class_features.shape[0] == number_of_samples_per_class, (
        f"Number of rows of concatenated class features {class_features.shape[0]} "
        f"does not match number of samples per class {number_of_samples_per_class}"
    )
    assert class_features.shape[1] == number_of_informative_features, (
        f"Number of columns of concatenated class features {class_features.shape[1]} "
        f"does not match number of informative features {number_of_informative_features}"
    )

    if distribution == "lognormal":
        class_features = transform_normal_distributed_class_features_to_lognormal_distribution(class_features)

    if plot_distribution:
        print("Distribution of informative class features:")
        plot_distribution_of_class_features_for_single_class(class_features, path_to_save_pdf=path_to_save_pdf)

    return class_features


def generate_pseudo_class(
    number_of_samples_per_class: int, number_of_pseudo_class_features: int, number_of_classes: int = 2
) -> ndarray:
    """Generate a pseudo class with the given number of samples per class and the given number of pseudo class features.

    The pseudo class is generated by concatenating the given number of classes with the given number of
    samples per class. ``number_of_classes`` distinct classes are generated and shuffled to create the pseudo class
    without relation to the class labels.

    Args:
        number_of_samples_per_class: Number of samples per class.
        number_of_pseudo_class_features: Number of pseudo class features.
        number_of_classes: Number of classes to generate.

    Returns:
        Numpy array of the given shape with pseudo class features.
    """
    rng = np.random.default_rng()
    simulated_classes = []
    for i in range(number_of_classes):
        # generate normal distributed random data
        simulated_class = rng.normal(
            loc=0,
            scale=1,
            size=(number_of_samples_per_class, number_of_pseudo_class_features),
        )
        # shift class to set effect size between classes equal to two times the class number
        shifted_simulated_class = shift_class_to_enlarge_effectsize(simulated_class, i * 2)
        if not math.isclose(np.mean(shifted_simulated_class), i * 2, abs_tol=0.15):
            logging.info(
                f"Mean of shifted class within pseudo classes {str(np.mean(shifted_simulated_class))} "
                f"differs from expected mean {str(i * 2)}."
            )
        assert shifted_simulated_class.shape[0] == number_of_samples_per_class, (
            f"Number of rows of simulated pseudo class element {shifted_simulated_class.shape[0]} "
            f"does not match number of samples per class {number_of_samples_per_class}"
        )
        assert shifted_simulated_class.shape[1] == number_of_pseudo_class_features, (
            f"Number of columns of simulated pseudo class element {shifted_simulated_class.shape[1]} "
            f"does not match number of pseudo class features {number_of_pseudo_class_features}"
        )
        simulated_classes.append(shifted_simulated_class)

    # concatenate classes
    pseudo_class = np.concatenate(simulated_classes, axis=0)
    assert pseudo_class.shape[0] == number_of_samples_per_class * number_of_classes, (
        f"Number of rows of pseudo class {pseudo_class.shape[0]} "
        f"does not match number of samples per class {number_of_samples_per_class} for {number_of_classes} classes"
    )
    assert pseudo_class.shape[1] == number_of_pseudo_class_features, (
        f"Number of columns of pseudo class {pseudo_class.shape[1]} "
        f"does not match number of pseudo class features {number_of_pseudo_class_features}"
    )

    # shuffle classes to create pseudo class
    rng.shuffle(pseudo_class)

    return pseudo_class


def generate_random_features(number_of_samples: int, number_of_random_features: int) -> ndarray:
    """Generate random noise or unrelated features.

    Args:
        number_of_samples: Number of rows of generated data.
        number_of_random_features: Number of columns of generated data.

    Returns:
        Numpy array of the given shape with normal distributed random numbers.
    """
    rng = np.random.default_rng()
    random_features_np = rng.random(
        size=(number_of_samples, number_of_random_features),
    )
    return random_features_np


def generate_artificial_classification_data(
    generated_classes_list: list[np.ndarray],
    number_of_samples_per_class: int,
    class_labels_list: list[int] = None,
    number_of_random_features: int = 0,
    number_of_pseudo_class_features: int = 0,
    number_of_pseudo_classes: int = 2,
) -> pd.DataFrame:
    """Generate artificial classification data.

    Args:
        generated_classes_list: List of labeled classes.
        number_of_samples_per_class: Number of samples per class.
        class_labels_list: List of class labels.
        number_of_random_features: Number of random features to generate.
        number_of_pseudo_class_features: Number of pseudo class features to generate.
        number_of_pseudo_classes: Number of pseudo classes to generate.

    Returns:
        Pandas DataFrame of the given shape with concatenated class features. Features are labeled as follows:

            - label: class label
            - bm: biomarker
            - pc: pseudo class
            - rf: random features
    """
    # check if number of samples per class is greater than zero
    if not number_of_samples_per_class > 0:
        raise ValueError("Number of samples per class must be greater than zero.")

    if not generated_classes_list:
        raise ValueError("Generated classes list is empty.")

    if not number_of_random_features >= 0:
        raise ValueError("Number of random features must be positive or zero.")

    if not number_of_pseudo_class_features >= 0:
        raise ValueError("Number of pseudo class features must be positive or zero.")

    # generate label as first entry for column names
    column_names = ["label"]

    # generate column names for class features (bm = biomarker)
    for column_index in range(generated_classes_list[0].shape[1]):
        column_names.append(f"bm_{column_index}")

    # append label to class features
    for i, class_features_np in enumerate(generated_classes_list):
        # check if class features are empty
        if class_features_np.shape[0] == 0:
            warnings.warn(f"Class {str(i)} is empty")

        # append label to class features
        if class_labels_list is not None:
            class_features_np = np.concatenate(
                (np.full((class_features_np.shape[0], 1), class_labels_list[i]), class_features_np), axis=1
            )
            # check of all elements of first column of class_features_data_array
            # are equal to the corresponding class label
            assert np.all(class_features_np[:, 0] == class_labels_list[i])
        else:
            class_features_np = np.concatenate((np.full((class_features_np.shape[0], 1), i), class_features_np), axis=1)
            # check of all elements of first column of class_features_data_array are equal to i (class label)
            assert np.all(class_features_np[:, 0] == i)
        generated_classes_list[i] = class_features_np

    # concatenate all class features
    class_data_np = np.concatenate(generated_classes_list, axis=0)
    assert len(column_names) == class_data_np.shape[1], (
        f"Number of column names {len(column_names)} " f"does not match number of columns {class_data_np.shape[1]}"
    )

    # concatenate all data components for the artificial classification data
    data_components_list = [class_data_np]

    # check if number of pseudo class features is greater than zero
    if number_of_pseudo_class_features > 0:
        # generate pseudo class
        pseudo_class = generate_pseudo_class(
            number_of_samples_per_class=number_of_samples_per_class,
            number_of_pseudo_class_features=number_of_pseudo_class_features,
            number_of_classes=number_of_pseudo_classes,
        )
        assert pseudo_class.shape[0] == class_data_np.shape[0], (
            f"Number of rows {pseudo_class.shape[0]} "
            f"does not match number of rows of classes {class_data_np.shape[0]}"
        )
        assert pseudo_class.shape[1] == number_of_pseudo_class_features, (
            f"Number of columns of generated pseudo class {pseudo_class.shape[1]} "
            f"does not match number of pseudo class features {number_of_pseudo_class_features}"
        )
        data_components_list.append(pseudo_class)

        # add column names for pseudo class
        for column_name in range(pseudo_class.shape[1]):
            column_names.append(f"pc_{column_name}")

    # check if number of random features is greater than zero
    if number_of_random_features > 0:
        # generate random features
        random_features_np = generate_random_features(
            number_of_samples=class_data_np.shape[0], number_of_random_features=number_of_random_features
        )
        data_components_list.append(random_features_np)

        # add column names for random features
        for column_name in range(random_features_np.shape[1]):
            column_names.append(f"rf_{column_name}")

    # concatenate class features, pseudo class and random features
    artificial_classification_data_np = np.concatenate(data_components_list, axis=1)
    assert len(column_names) == artificial_classification_data_np.shape[1], (
        f"Number of column names {len(column_names)} "
        f"does not match number of columns {artificial_classification_data_np.shape[1]}"
    )
    assert artificial_classification_data_np.shape[0] == class_data_np.shape[0], (
        f"Number of rows {artificial_classification_data_np.shape[0]} "
        f"does not match number of rows {class_data_np.shape[0]}"
    )
    assert np.all(
        artificial_classification_data_np[:, 0] == class_data_np[:, 0]
    ), "First column of artificial classification data does not match first column of class data"
    assert np.all(
        artificial_classification_data_np[:, 1 : class_data_np.shape[1]] == class_data_np[:, 1:]
    ), "Class features of artificial classification data do not match the original class features"
    assert (
        artificial_classification_data_np.shape[1]
        == class_data_np.shape[1] + number_of_pseudo_class_features + number_of_random_features
    ), (
        f"Number of columns {artificial_classification_data_np.shape[1]} "
        f"does not match sum of label, class features, pseudo class features and random features "
        f"{class_data_np.shape[1] + number_of_pseudo_class_features + number_of_random_features}"
    )
    # create pandas dataframe
    artificial_classification_data_df = pd.DataFrame(artificial_classification_data_np, columns=column_names)

    return artificial_classification_data_df


def find_perfectly_separated_features(list_of_feature_values_per_class: list[np.ndarray]) -> list[int]:
    """Find perfectly separated features in the given data.

    A feature is perfectly separated if all values of this feature are smaller or greater than the corresponding
    values of the same feature in all other classes. This function checks all features of all classes and returns
    a list of indices of perfectly separated features. The indices correspond to the columns of the given data.
    The label must be excluded from the input data.

    Args:
        list_of_feature_values_per_class: List of feature values per class. Each element of the list is a numpy array
            with shape (number_of_samples_per_class, number_of_features). The number of columns or features must be
            equal for each element of the given list. Each element of the list corresponds to a class. Note that the
            target/ labels of the given data must be excluded from the input data.

    Returns:
        List of indices of perfectly separated features.
    """
    # check if number of classes is greater than one
    if not len(list_of_feature_values_per_class) > 1:
        raise ValueError("Number of classes must be greater than one.")

    # check if number of features is greater than zero
    if not len(list_of_feature_values_per_class[0]) > 0:
        raise ValueError("Number of features must be greater than zero.")

    # check if number of features is equal in each class
    for i in range(len(list_of_feature_values_per_class) - 1):
        if not list_of_feature_values_per_class[i + 1].shape[1] == list_of_feature_values_per_class[i].shape[1]:
            raise ValueError("Number of features must be equal in each class.")

    # check which features perfectly separate all classes
    perfectly_separating_features = []

    # iterate over all features
    for i in range(list_of_feature_values_per_class[0].shape[1]):
        # iterate over all classes
        for j in range(len(list_of_feature_values_per_class) - 1):
            # check if all features of class j are smaller or greater than the corresponding features of class j+1
            if np.all(
                list_of_feature_values_per_class[j][:, i] < list_of_feature_values_per_class[j + 1][:, i]
            ) or np.all(list_of_feature_values_per_class[j][:, i] > list_of_feature_values_per_class[j + 1][:, i]):
                # print(f"Feature {i} perfectly separates class {j} and class {j + 1}")
                perfectly_separating_features.append(i)
    return perfectly_separating_features


def drop_perfectly_separated_features(
    list_of_perfectly_separated_features: list[int], data_df: pd.DataFrame, target_column_name: str | None = None
) -> pd.DataFrame:
    """Drop the perfectly separated informative features from the given data.

    Args:
        list_of_perfectly_separated_features: List of indices of perfectly separated features.
        data_df: Dataframe containing the data to drop the perfectly separated features from.
        target_column_name: Name of the target column (label) in the dataframe. If None, the first column is assumed to
            be the target column.
    Returns:
        Dataframe with dropped perfectly separated features.
    """
    # check if number of features is greater than zero
    if not len(list_of_perfectly_separated_features) > 0:
        raise ValueError("Number of perfectly separated features must be greater than zero.")

    if not len(data_df.columns) > len(list_of_perfectly_separated_features):
        raise ValueError("Number of columns in dataframe must be greater than number of perfectly separated features.")

    # select columns to drop by index
    if target_column_name is not None:
        if target_column_name not in data_df.columns:
            raise ValueError(f"Target column name {target_column_name} not found in dataframe columns.")
        feature_names = data_df.columns.drop([target_column_name])
        assert target_column_name not in feature_names, "Feature names must not include 'target_column_name'"
    else:
        feature_names = data_df.columns[1:]
    columns_to_drop = feature_names[list_of_perfectly_separated_features]
    for i, column_name in enumerate(columns_to_drop):
        assert (
            str(list_of_perfectly_separated_features[i]) in column_name
        ), f"Column name {column_name} must include given index {list_of_perfectly_separated_features[i]} as string"
        # only remove informative features
        if "bm" not in column_name:
            # drop column_name from columns_to_drop if it is not a biomarker
            columns_to_drop.remove(column_name)
        assert "bm" in column_name, "Column names must include 'bm' for biomarker"

    # drop perfectly separated features
    data_df = data_df.drop(columns_to_drop, axis=1)

    # check if all columns were dropped
    for dropped_column in columns_to_drop:
        assert dropped_column not in data_df.columns, f"Perfectly separated feature {dropped_column} was not dropped"

    if target_column_name is not None:
        # check if target column is still the DataFrame
        assert target_column_name in data_df.columns, (
            f"Target column {target_column_name} was dropped. "
            f"Please check if the target column is still in the DataFrame."
        )
    return data_df


def plot_correlated_cluster(
    feature_cluster: ndarray,
    correlation_method: Literal["pearson", "kendall", "spearman"] = "spearman",
    show_values: bool = True,
    path_to_save_pdf="",
) -> None:
    """Visualize the given cluster of correlated features.

    Args:
        feature_cluster: The cluster of correlated features to visualize.
        correlation_method: Method to calculate the correlation. Possible values are "pearson", "kendall" and "spearman".
        show_values: Show the correlation values in the visualization.
        path_to_save_pdf: Path to save the visualization as pdf.
    """
    sns.set_theme(style="white")
    data_df = pd.DataFrame(feature_cluster)
    corr = data_df.corr(method=correlation_method)

    print("min absolute correlation: " + str(np.min(np.abs(corr))))

    if show_values:
        sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2g")
    else:
        sns.heatmap(corr, cmap="Blues")

    if path_to_save_pdf != "":
        pyplot.savefig(path_to_save_pdf, dpi=300, format="pdf")

    pyplot.show()


def plot_distribution_of_class_features_for_single_class(
    class_features_np: ndarray, class_label: int = None, path_to_save_pdf=""
) -> None:
    """Visualize the histogram of the given array corresponding to the class features.

    Args:
        class_features_np: Class features to visualize.
        class_label: Label of the corresponding class.
        path_to_save_pdf: Path to save the visualization as pdf.
    """
    sns.set_theme(style="white")
    if class_label is not None:
        # remove label column
        class_plot_pd = pd.DataFrame(class_features_np[:, 1:].flatten(), columns=["class" + str(class_label)])
    else:
        class_plot_pd = pd.DataFrame(class_features_np.flatten(), columns=["class"])

    sns.histplot(data=class_plot_pd)

    if path_to_save_pdf != "":
        pyplot.savefig(path_to_save_pdf, dpi=300, format="pdf")

    pyplot.show()


def plot_distributions_of_all_classes(class_features_list: list[np.ndarray], path_to_save_pdf="") -> None:
    """Visualize the histogram of the given list of unlabeled arrays corresponding to the classes.

    Args:
        class_features_list: List of (unlabeled) classes to visualize.
        path_to_save_pdf: Path to save the visualization as pdf.
    """
    sns.set_theme(style="white")
    class_plot_pd = pd.DataFrame()
    for i, class_features_np in enumerate(class_features_list):
        class_plot_pd["class" + str(i + 1)] = class_features_np[:, 1:].flatten()

    sns.histplot(data=class_plot_pd)

    if path_to_save_pdf != "":
        pyplot.savefig(path_to_save_pdf, dpi=300, format="pdf")
    pyplot.show()


def plot_correlation_between_classes(class_features_list: list[np.ndarray], path_to_save_pdf="") -> None:
    """Visualize the correlation between the given list of unlabeled arrays corresponding to the classes.

    Args:
        class_features_list: List of (unlabeled) classes to visualize.
        path_to_save_pdf: Path to save the visualization as pdf.
    """
    class_plot_pd = pd.DataFrame()
    assert len(class_features_list) == 2, "Only two classes are supported for correlation visualization"
    for i, class_features_np in enumerate(class_features_list):
        class_plot_pd["class" + str(i)] = class_features_np.flatten()

    if path_to_save_pdf != "":
        pyplot.savefig(path_to_save_pdf, dpi=300, format="pdf")

    sns.pairplot(class_plot_pd)
