# Copyright (c) 2023 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT
"""Tests for the artificial_data_generator_tools module.

The artificial_data_generator_tools module is part of the artificial-data-generator package.
It provides functions for use in Jupyter notebooks to generate artificial data for classification tasks.
"""
import unittest
import numpy as np
import pandas as pd

from synthetic_data_generator import artificial_data_generator_tools as adgt


class TestArtificialDataGenerator(unittest.TestCase):
    def test_generate_correlated_cluster_with_valid_inputs(self):
        result = adgt.generate_correlated_cluster(5, 100, 0.1, 0.9)
        self.assertEqual(result.shape, (100, 5))

    def test_generate_correlated_cluster_with_invalid_inputs(self):
        with self.assertRaises(ValueError):
            adgt.generate_correlated_cluster(-5, 100, 0.1, 0.9)

    def test_generate_normal_distributed_informative_features_for_one_class_with_valid_inputs(self):
        result = adgt.generate_normal_distributed_informative_features_for_one_class(100, 5, 0.5)
        self.assertEqual(result.shape, (100, 5))

    def test_generate_normal_distributed_informative_features_for_one_class_with_invalid_inputs(self):
        with self.assertRaises(ValueError):
            adgt.generate_normal_distributed_informative_features_for_one_class(100, -5, 0.5)

    def test_transform_normal_distributed_class_features_to_lognormal_distribution(self):
        input_data = np.random.normal(0, 1, (100, 5))
        result = adgt.transform_normal_distributed_class_features_to_lognormal_distribution(input_data)
        self.assertEqual(result.shape, input_data.shape)

    def test_shift_class_to_enlarge_effectsize(self):
        input_data = np.random.normal(0, 1, (100, 5))
        result = adgt.shift_class_to_enlarge_effectsize(input_data, 2)
        self.assertEqual(result.shape, input_data.shape)

    def test_generate_pseudo_class_with_valid_inputs(self):
        result = adgt.generate_pseudo_class(100, 5)
        self.assertEqual(result.shape, (200, 5))

    def test_generate_pseudo_class_with_invalid_inputs(self):
        with self.assertRaises(ValueError):
            adgt.generate_pseudo_class(100, -5)

    def test_generate_random_features(self):
        result = adgt.generate_random_features(100, 5)
        self.assertEqual(result.shape, (100, 5))

    def test_generate_artificial_classification_data_with_valid_inputs(self):
        input_data = [np.random.normal(0, 1, (100, 5)) for _ in range(5)]
        result = adgt.generate_artificial_classification_data(input_data, 100)
        self.assertEqual(result.shape, (500, 6))

    def test_generate_artificial_classification_data_with_invalid_inputs(self):
        with self.assertRaises(ValueError):
            input_data = [np.random.normal(0, 1, (100, 5)) for _ in range(5)]
            adgt.generate_artificial_classification_data(input_data, -100)

    def test_find_perfectly_separated_features_with_valid_inputs(self):
        class1 = np.array([[1, 2, 3], [4, 5, 6]])
        class2 = np.array([[7, 8, 9], [10, 11, 12]])
        result = adgt.find_perfectly_separated_features([class1, class2])
        self.assertEqual(result, [0, 1, 2])

    def test_find_perfectly_separated_features_with_no_separated_features(self):
        class1 = np.array([[1, 2, 3], [4, 5, 6]])
        class2 = np.array([[1, 2, 3], [4, 5, 6]])
        result = adgt.find_perfectly_separated_features([class1, class2])
        self.assertEqual(result, [])

    def test_find_perfectly_separated_features_with_empty_class(self):
        class1 = np.array([])
        class2 = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            adgt.find_perfectly_separated_features([class1, class2])

    def test_find_perfectly_separated_features_with_one_class(self):
        class1 = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            adgt.find_perfectly_separated_features([class1])

    def test_find_perfectly_separated_features_with_different_number_of_features(self):
        class1 = np.array([[1, 2, 3], [4, 5, 6]])
        class2 = np.array([[7, 8], [9, 10]])
        with self.assertRaises(ValueError):
            adgt.find_perfectly_separated_features([class1, class2])


class TestDropPerfectlySeparatedFeatures(unittest.TestCase):
    def setUp(self):
        self.data_df = pd.DataFrame({
            'label': [1, 1, 0, 0],
            'bm_0': [1, 2, 3, 4],
            'bm_1': [5, 6, 7, 8],
            'bm_2': [9, 10, 11, 12]
        })

    def test_drop_perfectly_separated_features(self):
        result = adgt.drop_perfectly_separated_features([1], self.data_df)
        self.assertEqual(result.columns.tolist(), ['label', 'bm_0', 'bm_2'])

    def test_zero_perfectly_separated_features(self):
        with self.assertRaises(ValueError):
            adgt.drop_perfectly_separated_features([], self.data_df)

    def test_less_columns_than_perfectly_separated_features(self):
        with self.assertRaises(ValueError):
            adgt.drop_perfectly_separated_features([0, 1, 2, 3], self.data_df)

    def test_target_not_in_data_df(self):
        data_df = self.data_df[['bm_0', 'bm_1', 'bm_2']]
        with self.assertRaises(ValueError):
            adgt.drop_perfectly_separated_features([1], data_df, target_column_name="label")


class TestBuildClass(unittest.TestCase):
    def setUp(self):
        self.number_of_samples_per_class = 30
        self.number_of_informative_features = 50
        self.scale = 1.0
        self.correlated_clusters_list = [np.random.normal(0, 1, (30, 5)), np.random.normal(0, 1, (30, 5))]

    def test_build_class(self):
        result = adgt.build_class(self.number_of_samples_per_class, self.number_of_informative_features, self.scale, self.correlated_clusters_list)
        self.assertEqual(result.shape, (self.number_of_samples_per_class, self.number_of_informative_features))

    def test_build_uncorrelated_class(self):
        result = adgt.build_class(self.number_of_samples_per_class, self.number_of_informative_features)
        self.assertEqual(result.shape, (self.number_of_samples_per_class, self.number_of_informative_features))

    def test_zero_samples_per_class(self):
        with self.assertRaises(ValueError):
            adgt.build_class(0, self.number_of_informative_features, self.scale, self.correlated_clusters_list)

    def test_zero_informative_features(self):
        with self.assertRaises(ValueError):
            adgt.build_class(self.number_of_samples_per_class, 0, self.scale, self.correlated_clusters_list)

    def test_zero_scale(self):
        with self.assertRaises(ValueError):
            adgt.build_class(self.number_of_samples_per_class, self.number_of_informative_features, 0, self.correlated_clusters_list)


if __name__ == "__main__":
    unittest.main()
