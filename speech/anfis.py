from __future__ import division
from itertools import product
from fuzzy.subsets import FuzzySet
from speech.utils import lse

import pprint
import sys
sys.path.append('../')


class BaseModel(object):

    def __init__(self, sets_size, prec_params, mem_func):
        self._prec = prec_params
        self._rule_set = self._create_rules(sets_size)

        if sum(sets_size) != len(prec_params):
            raise ValueError('Invalid number params and membership functions! \
                {} != {}'.format(sum(sets_size), len(prec_params)))

        self.subsets = [FuzzySet(mem_func) for i in sets_size]
        self._sets = sets_size
        for i in range(1, len(sets_size)):
            self._sets[i] = self._sets[i - 1] + sets_size[i]

    def _create_rules(self, sets_size):
        """ Create all combinations of the fuzzy labels, that is all the
        precedents rules.

        Parameters
        ----------
        stes_size : list of int
            A list with the number of labels in each fuzzy set

        Returns
        -------
        rules_combination : list of tuples
            A list of tuples with all combinations of the given fuzzy sets

        Raises
        ------
        ValueError
            On empty parameter, element or negative size
        """
        if sets_size is None:
            raise ValueError('Sets size is None!')
        elif None in sets_size:
            raise ValueError('Input contains None as size!')
        for size in sets_size:
            if size <= 0:
                raise ValueError('Invalid size ' + str(size))

        rules_set = [range(num_of_rules) for num_of_rules in sets_size]
        rules_combinations = [comb for comb in product(*rules_set)]
        return rules_combinations

    def _product_operation(self, fuzz_output):
        """ Execute a product operation with the fuzzy values from the first layer.
        That is also known as .. operation.

        Parameters
        ----------
        fuzz_output : list of list of double
            A list with the outputs of each label in the fuzzy sets

        Returns
        -------
        prod_outputs : list of double
            A list with the product of the outputs of each precedent rule
            combination.

        Raises
        ------
        ValueError
            On None argument of None element on argument
        """
        if fuzz_output is None:
            raise ValueError('Fuzz outputs cannot be None!')
        elif None in fuzz_output:
            raise ValueError('Fuzz outputs has a None element!')

        prod_outputs = []
        for rule in self._rule_set:
            prod = 1.0
            for output, label in zip(fuzz_output, rule):
                prod *= output[label]
            prod_outputs.append(prod)
        return prod_outputs

    def _min_operation(self, fuzz_output):
        """ Execute a min operation with the fuzzy values from the first layer.
        That is also known as .. operation.

        Parameters
        ----------
        fuzz_output : list of list of double
            A list with the outputs of each label in the fuzzy sets

        Returns
        -------
        minimum_outputs : list of double
            A list with the minimum output of each precedent rule combination

        Raises
        ------
        ValueError
            On None argument of None element on argument
        """
        if fuzz_output is None:
            raise ValueError('Fuzzy outputs cannot be None!')
        if None in fuzz_output:
            raise ValueError('Output has None element!')

        minimum_outputs = []
        for rule in self._rule_set:
            minimum = fuzz_output[0][rule[0]]
            for output, label in zip(fuzz_output, rule):
                if output[label] < minimum:
                    minimum = output[label]
            minimum_outputs.append(minimum)
        return minimum_outputs

    def learn_hybrid_online(self, data, threshold=1e-10, max_epochs=500):
        raise NotImplementedError()

    def _find_consequents(values, weights, new_line=False):
        raise NotImplementedError()

    def forward_pass(self, entries, expected, min_prod=False, new_row=False):
        raise NotImplementedError()

    def backward_pass(self):
        raise NotImplementedError()


class SugenoModel(BaseModel):
    pass


class MamdaniModel(BaseModel):
    pass


class TsukamotoModel(BaseModel):
    """ Class to represent an Artifical Neural Fuzzy Inference System which
    uses the Tsukamoto Fuzzy Inference System.
    """

    def __init__(self, sets_size, prec_params, prec_fun, cons_fun):
        """ Initialize a new Tsukamoto Fuzzy Inference System

        Parameters
        ----------
        sets_size : list of int
            The size of each fuzzy set in the precendents layer
        prec_params : 2D list of double
            A matrix with the precedent parameters to be used on the fuzzy sets
        prec_fun : MembershipFunction
            A MembershipFcuntion to be used on the precedent layer
        cons_fun : MembershipFunction
            A MembershipFcuntion to be used on the consequent layer. Note that,
            to be used in the Hybrid Learning method, this function must allow
            the system to have its parameters in evidence.
        """
        super(TsukamotoModel, self).__init__(sets_size, prec_params, prec_fun)
        self.cons_fun = cons_fun
        self.coef_matrix = []
        self.expected = []

    def _build_coefmatrix(self, values, weights, new_row=False):
        """ Updates the linear system as new equations are available

        Parameters
        ----------
        values : list of double
            The inputs to the fourth layer
        weights : list of double
            The outputs from the third layer
        new_row : boolean
            Defaults to False. If set to true a new equation is added to the
            system, otherwise the last equation is overwritten.
        """
        tmp_row = []
        for value, weight in zip(values, weights):
            tmp_row.extend(
                self.cons_fun.build_sys_row(value, weight)
            )
        if new_row or len(self.coef_matrix) == 0:
            self.coef_matrix.append(tmp_row)
        else:
            self.coef_matrix[-1] = tmp_row

    def _find_consequents(self, values, weights, expec_result, new_row=False):
        self._build_coefmatrix(values, weights, new_row)
        if new_row or len(self.expected) == 0:
            self.expected.append(expec_result)
        else:
            self.expected[-1] = expec_result
        result = lse(self.coef_matrix, self.expected)
        return [rs[0] for rs in result]

    def learn_hybrid_online(self, data, threshold=1e-10, max_epochs=500):
        for pair in data:
            epoch = 0
            while epoch < max_epochs:
                l1, l2, l5 = self.forward_pass(
                    pair[0], pair[1], False, epoch == 0
                )
                epoch += 1

    def forward_pass(self, entries, expected, min_prod=False, new_row=False):
        """ This method will compute the outputs from layers 1 to 4. The fourth
        layer will be calculated after finding the parameters with a Least
        Square Estimation.

        Parameters
        ----------
        entries : list of double
            The inputs to the ANFIS
        expected : double
            The expected value to the given parameters
        min_prod : boolean
            If set to True the min operation will be used to compute the layer
            2 outputs. Otherwise, the product operation will be used. Defaults
            to False.

        Returns
        ------
        layer_1 : list of double
            The outputs from layer one
        layer_2 : list of double
            The outputs from layer two
        layer_3 : list of double
            The outputs from layer three
        """
        # print 'Number of rules: ' + str(len(self._rule_set))
        layer_1 = []
        last_idx = 0
        for entry, subset, idx in zip(entries, self.subsets, self._sets):
            layer_1.append(subset.evaluate(entry, self._prec[last_idx:idx]))
            last_idx = idx

        layer_2 = []
        if min_prod:
            layer_2 = self._min_operation(layer_1)
        else:
            layer_2 = self._product_operation(layer_1)

        denom = sum(layer_2)
        layer_3 = [elm / denom for elm in layer_2]

        consequents = self._find_consequents(
            layer_2, layer_3, expected, new_row
        )

        cons_membership = []
        for i in range(len(layer_2)):
            init = i * 2
            end = init + 2
            mem_value = self.cons_fun.membership_degree(
                layer_2[i], *consequents[init:end]
            )
            cons_membership.append(mem_value)
        layer_4 = [weight * fi for weight, fi in zip(layer_3, cons_membership)]
        layer_5 = sum(layer_4)
        return layer_1, layer_2, layer_5

    def backward_pass(self, layer_1, layer_2, layer_5):
        pass
