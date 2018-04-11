from itertools import product
from fuzzy.subsets import FuzzySet

import sys
sys.path.append('../')


def _find_consequents(self):
    pass


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

    def forward_pass(self, pair):
        raise NotImplementedError()

    def backward_pass(self):
        raise NotImplementedError()


class SugenoModel(BaseModel):
    pass


class MamdaniModel(BaseModel):
    pass


class TsukamotoModel(BaseModel):
    """
    """
    def __init__(self, sets_size, prec_params, mem_func):
        super(TsukamotoModel, self).__init__(sets_size, prec_params, mem_func)

    def learn_hybrid_online(self, data, threshold=1e-10, max_epochs=500):
        pass

    def forward_pass(self, entries, expected, min_prod=False):
        """
        """
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

        return layer_1, layer_2
