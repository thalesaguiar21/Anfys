import anfys.neural.builder as builder
import numpy as np
import anfys.lse as regression
from itertools import product


_FEATURE_VECTOR_DIMENSION = 1
_DATA_LENGTH = 0
_CONS_MF_NUM = 2


class ANFIS:

    def __init__(self, subset_size):
        self.subset_size = subset_size
        self.qtd_rules = 0
        self.fuzzysets = []
        self.cons_params = []
        self.prem_params = []
        self.linsys_coefs = []
        self.linsys_resul = []
        self.prem_mf = None
        self.regressor = None

    def fit_by_hybrid_learn(self, inputs, outputs, max_epochs):
        builder.configure_model(inputs.shape[_FEATURE_VECTOR_DIMENSION])
        epoch = 1
        while epoch <= max_epochs:
            for entry, output in zip(inputs, outputs):
                self._full_forwardpass_hybrid_learn(entry, output)
            epoch += 1

    def _full_forwardpass_hybrid_learn(self, entry, output):
        raise NotImplementedError(
            'Call to _full_forwardpass_hybrid_learn in base class!')

    def _half_forward_pass(self, entry, output):
        # Forward inputs until the third layer
        layer1 = self._fuzzysets_membership_degrees(entry)
        layer2 = self._rules_fire_strength(layer1)
        layer3 = self._averaged_fire_strength(layer2)
        return layer1, layer2, layer3

    def _fuzzysets_membership_degrees(self, inputs):
        layer1 = np.zeros((self.l1_size(), self.qtd_mfs))
        param_range = np.arange(0, self.l1_size() + 1, self.qtd_mfs)
        for feat, subset, out in zip(inputs, self.sets, range(self.l1_size())):
            at, untill = param_range[out], param_range[out + 1]
            output = subset.evaluate(feat, self.prem_params[at:untill])
            layer1[out] = output
        return layer1

    def _rules_fire_strength(self, mdegrees):
        nodes_id = np.arange(self.qtd_mfs)
        qtd_sets = self.qtd_inputs
        # Create every combination for the given
        layer2 = []
        for mf in product(nodes_id, repeat=self.qtd_inputs):
            rule = [mdegrees[n_set, mf[n_set]] for n_set in range(qtd_sets)]
            layer2.append(np.prod(rule))
        return np.array(layer2)

    def _averaged_fire_strength(self, fire_strengths):
        total_strength = np.sum(fire_strengths)
        return [rstrength / total_strength for rstrength in fire_strengths]

    def _update_consequent_parameters(self, entry, output, weights):
        column_weights = np.array([weights]).T
        new_equation = column_weights.dot(entry)
        self.add_linsys_equation(new_equation)
        self.cons_params = regression.solve(
            self.linsys_coefs, self.linsys_resul)

    def _consequent_membership_degree(self):
        column_cons_params = self.cons_params.T
        self.linsys_coefs.dot(column_cons_params)

    def _prediction(self, consequent_memdegree):
        return np.sum(consequent_memdegree)

    def add_linsys_equation(self, coefs, result):
        self.linsys_coefs.append(coefs)
        self.linsys_resul.append(result)


class Sugeno(ANFIS):

    def __init__(self, subset_size):
        super().__init__(subset_size)

    def _full_forwardpass_hybrid_learn(self, entry, output):
        l1, l2, l3 = self._half_forward_pass(entry, output)
        self._update_cons_params(entry, output, l3)
        l4 = self._consequent_membership_degrees()
        l5 = self._prediction(l4)
        return l5
