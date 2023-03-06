import unittest
import numpy.testing as nptest
import linreg.weights as weights


class MyTestCase(unittest.TestCase):
    @staticmethod
    def test_weights_one_feature():
        weights_vect = weights.get_weights([[5.6, 6.5, 6.8, 6.9, 7.0, 7.4, 8.0, 8.3, 8.7, 9.0]],
                                           [5.0, 7.1, 8.4, 7.3, 7.8, 8.1, 7.4, 8.9, 9.0, 10.0])

        nptest.assert_array_almost_equal(weights_vect, [[-0.342, 1.11]])  # add assertion here


if __name__ == '__main__':
    unittest.main()
