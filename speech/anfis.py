from itertools import product
import sys
sys.path.append('../')


def _find_consequents(self):
    pass


class BaseModel:

    def __init__(self, sets_size, prec_params):
        self.__prec = prec_params
        self._rule_set = self._create_rules(sets_size)

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

    def learn_hybrid_online(self):
        raise NotImplementedError()

    def forward_pass(self):
        raise NotImplementedError()

    def backward_pass(self):
        raise NotImplementedError()


class SugenoModel(BaseModel):
    pass


class MamdaniModel(BaseModel):
    pass


class TsukamotoModel(BaseModel):
    pass
