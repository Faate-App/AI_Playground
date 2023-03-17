import unittest
import numpy.testing as nptest
import linreg.weights as weights


class MyTestCase(unittest.TestCase):
    @staticmethod
    def test_weights_one_feature():
        weights_vect = weights.get_weights([[5.6, 6.5, 6.8, 6.9, 7.0, 7.4, 8.0, 8.3, 8.7, 9.0]],
                                           [5.0, 7.1, 8.4, 7.3, 7.8, 8.1, 7.4, 8.9, 9.0, 10.0])

        nptest.assert_array_almost_equal(weights_vect, [[-0.342, 1.11]], 3)  # add assertion here

    @staticmethod
    def test_weights_two_features():
        weights_vect = weights.get_weights([[2.0, 1.5, 4.0, 5.0, 1.0, 3.2, 6.0, 2.5, 0.5, 4.3, 7.0, 0.1, 5.5, 6.2],
                                            [4.0, 2.25, 16.0, 25.0, 1.0, 10.24, 36.0, 6.25, 0.25, 18.49, 49.0, 0.01,
                                             30.25, 38.44]],
                                           [27.33, 28.20, 26.54, 21.24, 26.35, 25.88, 19.62, 29.69, 25.10, 25.14, 7.41, 20.10, 19.63, 15.36])

        nptest.assert_array_almost_equal(weights_vect, [[21.379, 5.361, -1.026]], 3)  # add assertion here


if __name__ == '__main__':
    unittest.main()
