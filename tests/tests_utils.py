from .context import utils
import unittest


class TestUtils(unittest.TestCase):

    def test_lev_dist_kitten(self):
        self.assertEqual(
            3.0, utils.levenshtein_distance('kitten', 'sitting'))

    def test_lev_dist_diff(self):
        self.assertEqual(
            6.0, utils.levenshtein_distance('kitten', ''))

    def test_lev_dist_equal(self):
        self.assertEqual(
            0.0, utils.levenshtein_distance('kitten', 'kitten'))

    def test_lev_dist_list(self):
        self.assertEqual(
            0.0, utils.levenshtein_distance(list('kitten'), 'kitten'))
