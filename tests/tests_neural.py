from .context import anfys
import anfys.neural.anfis as anfis
import anfys.fuzzy.mem_funcs as mfs
import unittest
import numpy as np


class TestTsukamoto(unittest.TestCase):

    def test_build(self):
        model = anfis.Tsukamoto(3, mfs.BellTwo())
        model.hybrid_learn(np.zeros((10, 30)), 0)

    def test_forward_pass(self):
        model = anfis.Tsukamoto(3, mfs.BellTwo())
        model.hybrid_learn(np.zeros((10, 10)), 2)
