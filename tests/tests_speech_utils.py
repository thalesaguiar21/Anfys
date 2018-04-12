from context import utils
import unittest2 as unittest


class TestUtils(unittest.TestCase):

    def test_lse_online(self):
        coef_matrix = [[2, 3, -1], [4, -1, 2]]
        rs_matrix = [5, -1]
        expected = [0.35087719, 1.23859648, -0.58245613]
        res = utils.lse(coef_matrix, rs_matrix, 10000000)
        for expected, curr in zip(expected, res):
            self.assertAlmostEqual(expected, curr[0])

    def test_lse_offline(self):
        """ Test for lse offline with the following linear system

        2x + 3y = 5
        4x -  y = -1

        The result is x = 0.1428571428571429 and y = 1.571428571428571
        """
        coef_matrix = [[2, 3], [4, -1]]
        rs_matrix = [5, -1]
        expected = [0.1428571428571429, 1.571428571428571]
        res = utils.lse(coef_matrix, rs_matrix, 10000000)
        for expected, curr in zip(expected, res):
            self.assertAlmostEqual(expected, curr[0])
