# Copyright (c) 2022 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Example for usage of generator for artificial data synthetic_data_generator."""

from synthetic_data_generator import synthetic_data_generator, visualizer


params_dict = {
    "number_of_relevant_features": 12,
    "number_of_pseudo_class_features": 2,
    "random_features": {"number_of_features": 10, "distribution": "lognormal", "scale": 1, "mode": 0},
    "classes": {
        1: {
            "number_of_samples": 15,
            "distribution": "lognormal",
            "mode": 3,
            "scale": 1,
            "correlated_features": {
                1: {"number_of_features": 4, "correlation_lower_bound": 0.7, "correlation_upper_bound": 1},
                2: {"number_of_features": 4, "correlation_lower_bound": 0.7, "correlation_upper_bound": 1},
                3: {"number_of_features": 4, "correlation_lower_bound": 0.7, "correlation_upper_bound": 1},
            },
        },
        2: {"number_of_samples": 15, "distribution": "normal", "mode": 1, "scale": 2, "correlated_features": {}},
        3: {"number_of_samples": 15, "distribution": "normal", "mode": -10, "scale": 2, "correlated_features": {}},
    },
    "seed": None,
    "path_to_save_csv": "your_path_to_save.csv",
    "path_to_save_feather": "",
    "path_to_save_meta_data": "your_path_to_save_params_dict.yaml",
    "shuffle_features": False,
}

# generate artificial data as Dataframe
data_df = artificial_data_generator.generate_artificial_classification_data(params_dict)

# visualize generated correlations and class distributions
visualizer.visualize(data_df, params_dict)
