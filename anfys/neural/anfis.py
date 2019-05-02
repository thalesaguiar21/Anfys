import numpy as np
import itertools
from anfys.fuzzy.subsets import FuzzySet
from anfys.fuzzy.mem_funcs import PiecewiseLogit, BellTwo
import pdb


_INPUT_DIMENSION = 1
_DATA_LENGTH = 0
_CONS_MF_NUM = 2


class Tsukamoto:

    def __init__(self, fuzzy_set_size, prec_mem_function):
        self.qtd_rules = 0
        self.sets = []
        self.qtd_mfs = fuzzy_set_size
        self.prem_mf = BellTwo()
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
        self.qtd_rules = self.qtd_inputs ** self.qtd_mfs
        self.sets = [FuzzySet(self.prem_mf) for _ in range(self.qtd_inputs)]
        self.build_prem_params()
        self.cons_params = np.zeros((self.qtd_rules, _CONS_MF_NUM))

    def build_prem_params(self, stdev=1.0):
        stdevs = np.ones(self.l1_size()) * stdev
        means = np.linspace(-1.0, 1.0, self.qtd_mfs)
        means = np.array(means.tolist() * self.qtd_inputs)
        self.prem_params = np.vstack((stdevs, means)).T

    def _forward_pass(self, entry):
        inputs = entry[:-1]
        layer1 = self.layer1_output(inputs)
        self.layer2_output(layer1)

    def l1_size(self):
        return self.qtd_mfs * self.qtd_inputs

    def layer1_output(self, inputs):
        layer1 = []
        i = 0
        param_range = np.arange(0, self.l1_size() + 1, self.qtd_mfs)
        for feat, subset in zip(inputs, self.sets):
            at, untill = param_range[i], param_range[i + 1]
            output = subset.evaluate(feat, self.prem_params[at:untill])
            layer1.append(output)
            i += 1
        return np.array(layer1)

    def layer2_output(self, inputs):
        nodes_id = np.arange(self.qtd_mfs)
        # Create every combination for the given
        for rule in itertools.product(nodes_id, repeat=self.qtd_inputs):
            pass

    def _backward_pass(self):
        pass
