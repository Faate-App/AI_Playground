import numpy as np


def sum_features_multiplied(features_mat):
    return np.matmul(features_mat, np.matrix.transpose(features_mat))


def get_weights(features_mat, y):
    xtx_inverse = np.invert(sum_features_multiplied(features_mat))
    xtranspose = np.transpose(features_mat)
    weights_vect = np.matmul(np.matmul(xtx_inverse, xtranspose), y)
    return weights_vect

