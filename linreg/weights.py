import numpy as np
import numpy.linalg as nl


def sum_features_multiplied(features_mat):
    return np.matmul(features_mat, np.matrix.transpose(features_mat))


def get_weights(features_mat, y):
    xarr = np.array(features_mat, ndmin=2)
    yarr = np.array(y)

    xtx_inverse = nl.inv(sum_features_multiplied(xarr))
    xtranspose = np.matrix.transpose(xarr)

    inverse_transpose_mult = np.matmul(xtranspose, xtx_inverse)
    weights_vect = np.matmul(yarr, inverse_transpose_mult)

    return weights_vect
