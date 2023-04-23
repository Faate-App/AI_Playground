import math

import numpy as np
import copy
import model_errors as m_err


def k_cross_validation(model_function, k, dataset_matrix):  # TODO
    exact_y_arr = dataset_matrix[len(dataset_matrix) - 1]
    indices_array = np.random.rand((k,))
    possible_indices = range(0, len(dataset_matrix))

    while len(possible_indices) <= len(exact_y_arr):
        matrix_copy = copy.deepcopy(dataset_matrix)
        test_feature_values = []

        for index in indices_array:
            if index not in possible_indices:
                np.delete(possible_indices, index)


def mdl(model_func, model_complexity, dataset_size, exact_y_arr):
    return dataset_size * math.log(m_err.mse(model_func, exact_y_arr)) + model_complexity * math.log(dataset_size)
