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


print(get_weights([5.6, 6.5, 6.8, 6.9, 7.0, 7.4, 8.0, 8.3, 8.7, 9.0],
                  [5.0, 7.1, 8.4, 7.3, 7.8, 8.1, 7.4, 8.9, 9.0, 10.0]))
