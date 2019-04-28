from .context import lse
import unittest
import numpy as np


class TestMatrix(unittest.TestCase):

    def test_matricial_overdetermined(self):
        coef_matrix = np.array([[1, -1], [1, 1], [2, 1]])
        rs_matrix = np.array([2, 4, 8])
        expected = [23 / 7, 8 / 7]
        res = lse.Matricial().solve(coef_matrix, rs_matrix)
        self.assertSequenceAlmostEqual(self, expected, res)

    def test_matricial_determined(self):
        coef_matrix = np.array([[2, 3], [4, -1]])
        rs_matrix = np.array([5, -1])
        expected = [0.1428571428571429, 1.571428571428571]
        res = lse.Matricial().solve(coef_matrix, rs_matrix)
        self.assertSequenceAlmostEqual(self, expected, res)

    def test_matricial_underdetermined(self):
        pass


class TestRecursive(unittest.TestCase):

    def test_recursive_determined(self):
        coef_matrix = np.array([[2, 3, 2], [1, 3, 2], [1, 2, 2]])
        rs_matrix = np.array([12, 13, 11])
        expected = [12.002992517583628, 12.998997523491125, 10.996013446352867]
        res = lse.Recursive(1.0, 1000).solve(coef_matrix, rs_matrix)
        self.assertSequenceAlmostEqual(self, expected, res)

    def test_recursive_overdetermined(self):
        coef_matrix = np.array([[1, -1], [1, 1], [2, 1]])
        rs_matrix = np.array([2, 4, 8])
        expected = [23 / 7, 8 / 7]
        res = lse.Recursive(1.0, 1000).solve(coef_matrix, rs_matrix)
        self.assertSequenceAlmostEqual(self, expected, res)

def assertSequenceAlmostEqual(testcase, seq1, seq2, tolerance=7):
    for s1, s2 in zip(seq1, seq2):
        testcase.assertAlmostEqual(s1, s2, tolerance)


class TestClipping(unittest.TestCase):

    def when_values_are(self, value, lower, upper):
        self.value = value
        self.lower = lower
        self.upper = upper

    def do_clip(self):
        return lse.clip(self.value, self.lower, self.upper)

    def test_clipping_inside(self):
        when_values_are(2, 1, 100)
        clippedvalue = self.do_clip()
        self.assertEqual(clippedvalue, value)

    def test_clipping_outside_max(self):
        when_values_are(1000, 1, 100)
        clippedvalue = self.do_clip()
        self.assertEqual(clippedvalue, maxvalue)

    def test_clipping_close_max(self):
        when_values_are(100 - 1e-10, 1, 100)
        clippedvalue = self.do_clip()

    def test_clipping_outside_min(self):
        when_values_are(-10, 1, 100)
        clippedvalue = self.do_clip()
        self.assertEqual(clippedvalue, minvalue)

    def test_clipping_close_min(self):
        when_values_are(1 - 1e-10, 1, 100)
        clippedvalue = self.do_clip()
        self.assertEqual(clippedvalue, minvalue)
