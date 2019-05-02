import numpy as np
import itertools
import pdb
from anfys.fuzzy.subsets import FuzzySet
from anfys.fuzzy.mem_funcs import PiecewiseLogit


_INPUT_DIMENSION = 1
_DATA_LENGTH = 0


class Tsukamoto:

    def __init__(self, fuzzy_set_size, prec_mem_function):
        self.rules = []
        self.sets = []
        self.qtd_mfs = fuzzy_set_size
        self.prem_mf = prec_mem_function
        self.cons_mf = PiecewiseLogit()
        self.prem_params = []
        self.cons_params = []
        self.qtd_inputs = 0

    def hybrid_learn(self, entries, max_epochs):
        self._build_archtecture(entries.shape)
        epoch = 1
        while epoch <= max_epochs:
            for entry in entries:
                self._forward_pass(entry)
            epoch += 1

    def _build_archtecture(self, dimensions):
        self.qtd_inputs = dimensions[_INPUT_DIMENSION]
        self.sets = [FuzzySet(self.prem_mf) for _ in range(self.qtd_inputs)]
        self.build_prem_params()

    def build_prem_params(self, stdev=1.0):
        stdevs = np.ones(self.l1_size()) * stdev
        means = np.linspace(-1.0, 1.0, self.qtd_mfs)
        means = np.array(means.tolist() * self.qtd_inputs)
        self.prem_params = np.vstack((stdevs, means)).T

    def _forward_pass(self, entry):
        inputs, out = entry[:-1], entry[-1]
        layer1 = []
        i = 0
        param_range = np.arange(0, self.l1_size() + 1, self.qtd_mfs)
        for feat, subset in zip(entry, self.sets):
            params = self.prem_params[param_range[i]:param_range[i + 1]]
            layer1.append(subset.evaluate(feat, params))
            i += 1

    def l1_size(self):
        return self.qtd_mfs * self.qtd_inputs

    def _backward_pass(self):
        pass


def create_rules(fuzzy_set_size, qtd_inputs):
    nodes_id = np.arange(fuzzy_set_size)
    rules = [x for x in itertools.product(nodes_id, repeat=qtd_inputs)]
    return np.array(rules)
