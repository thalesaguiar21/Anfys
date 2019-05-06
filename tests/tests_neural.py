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


class TestSugeno(unittest.TestCase):

    def when_model_qtd_of_mf_is(self, qtd_mf):
        self.model = anfis.Sugeno(qtd_mf)

    def test_setup_arch(self):
        self.when_model_qtd_of_mf_is(3)
