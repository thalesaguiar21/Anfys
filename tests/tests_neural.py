from .context import anfys
import anfys.neural.anfis as anfis
import unittest
import numpy as np


class TestTsukamoto(unittest.TestCase):

    def when_model_mfs_is(self, qtd_mf):
        self.model = anfis.Tsukamoto(qtd_mf)

    def hybrid_learn_for_data_shape(self, nsamp, nfeat):
        inputs = np.zeros((nsamp, nfeat))
        out = np.zeros(nsamp)
        return self.model.hybrid_learn(inputs, out, 1)

    def test_forward_pass(self):
        model = anfis.Tsukamoto(2)
        model.hybrid_learn(np.zeros((10, 3)), np.zeros(10), 1)

    def test_8_rules(self):
        self.when_model_mfs_is(2)
        _, l2, _ = self.hybrid_learn_for_data_shape(10, 3)
        self.assertEqual(l2.size, 8)

    def test_qtdrules_for_0inputs(self):
        self.when_model_mfs_is(2)
        _, l2, _ = self.hybrid_learn_for_data_shape(0, 0)
        self.assertEqual(l2.size, 0)


class TestSugeno(unittest.TestCase):

    def when_model_qtd_of_mf_is(self, qtd_mf):
        self.model = anfis.Mamdani(qtd_mf)

    def test_setup_arch(self):
        self.when_model_qtd_of_mf_is(3)
