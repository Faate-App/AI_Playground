import numpy as np


# get weights coeffs for a model according to matrix formula
def get_weights(features_mat, y):
    # verticalization of vectors
    yarr = np.array(y).transpose()
    xarr = np.matrix(features_mat)
    xarr = np.insert(arr=xarr, obj=0, values=np.ones(yarr.shape), axis=0).transpose()

    x_transpose = xarr.transpose()

    xtx = x_transpose.dot(xarr)

    xtx_inv = np.linalg.inv(xtx)

    weights = xtx_inv.dot(x_transpose).dot(yarr)

    return weights
