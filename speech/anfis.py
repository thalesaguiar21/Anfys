from __future__ import division
from itertools import product
from fuzzy.subsets import FuzzySet
from speech.utils import lse_online
from math import sqrt

import pdb
import time
import numpy as np
import sys
sys.path.append('../')


class BaseModel(object):

    def __init__(self, sets_size, prec_params, mem_func):
        self._prec = np.array(prec_params)
        self._rule_set = self._create_rules(np.array(sets_size))

        if sum(sets_size) != len(prec_params):
            raise ValueError('Invalid number params and membership functions! \
                {} != {}'.format(sum(sets_size), len(prec_params)))

        self.subsets = [FuzzySet(mem_func) for i in sets_size]
        self._sets = sets_size
        self._errors = []
        for i in xrange(1, len(sets_size)):
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
        return np.array([comb for comb in product(*rules_set)])

    def _product_operation(self, fuzz_output):
        """ Execute a product operation with the fuzzy values from the first
        layer. That is also known as t-norm operation.

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
        return np.array(prod_outputs)

    def _min_operation(self, fuzz_output):
        """ Execute a min operation with the fuzzy values from the first layer.
        That is also known as t-norm operation.

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
            for i in xrange(rule.size):
                if fuzz_output[i][rule[i]] < minimum:
                    minimum = fuzz_output[i][rule[i]]
            minimum_outputs.append(minimum)
        return np.array(minimum_outputs)

    def learn_hybrid_online(self, data, threshold=1e-10, max_epochs=500):
        raise NotImplementedError()

    def _find_consequents(values, weights, new_line=False):
        raise NotImplementedError()

    def forward_pass(self, entries, expected, min_prod=False, new_row=False):
        raise NotImplementedError()

    def backward_pass(self, layer_1, layer_2, layer_5, error):
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
        col = self._rule_set.shape[0] * len(cons_fun.parameters)
        self.coef_matrix = np.empty((0, col))
        self.expected = np.empty((0, 1))

    def _build_coefmatrix(self, values, weights, expec, newrow=False):
        """ Updates the linear system as new equations are available

        Parameters
        ----------
        values : list of double
            The inputs to the fourth layer
        weights : list of double
            The outputs from the third layer
        expec : double
            The expected result for this line of the system
        newrow : boolean
            Defaults to False. If set to true a new equation is added to the
            system, otherwise the last equation is overwritten.
        """
        tmp_row = np.empty((1, 0))
        for value, weight in zip(values, weights):
            row_term = self.cons_fun.build_sys_term(value, weight)
            tmp_row = np.append(tmp_row, row_term)
        if newrow or self.coef_matrix.size == 0:
            self.coef_matrix = np.vstack([self.coef_matrix, tmp_row])
            self.expected = np.append(self.expected, expec)
        else:
            # Replace the last line with the new parameters
            self.coef_matrix[-1, :] = tmp_row
            self.expected[-1] = expec

    def _find_consequents(self, values, weights, expec_result, newrow=False):
        """ Handle the search for consequent parameters by using LSE and
        expanding the linear system.

        Parameters
        ----------
        values : list of double
            The inputs to the consequent membership functions.
        weights : list of double
            The weights used in the outputs of the fourth layer. In this model
            the values from the third layer are used as weights.
        expec_result : double
            The result expected for the given entries
        newrow : boolean
            Inform that this epoch is training a new pair, and that should be
            added to the linear system. Defaults to false.
        """
        self._build_coefmatrix(values, weights, expec_result, newrow)
        result = lse_online(self.coef_matrix, self.expected, 0.9999, 1000.)
        return [rs[0] for rs in result]

    def learn_hybrid_online(
            self, data, tol=0.000001, max_epochs=500, prod=False):
        """ Train the ANFIS with the given data pairs.

        Parameters
        ----------
        data : list of pars of list and integer
            The training data set. For instance [([2, 3, 4], 10)], where 10 is
            the expected value for [2, 3, 4] entry.
        tol : double
            The error tolerance. Defaults to 1e-10
        max_epochs : integer
            The maximum number of epochs for each pair. Defaults to 500
        prod : boolean
            Set this to True to use product T-Norm, otherwise the Min T-Norm
            will be used. Defaults to False.
        """
        for pair in data:
            if len(pair[0]) != len(self._sets):
                raise ValueError('Number of inputs must match number of sets!')

        for pair in data:
            epoch = 0
            print 'Running pair --> ' + str(pair)
            while epoch < max_epochs:
                newrow = epoch == 0
                l1, l2, l3, l5 = self.forward_pass(
                    pair[0], pair[1], prod, newrow
                )
                # Compute the iteration error
                error = pair[1] - l5
                # print '{}-th epoch | {} - {:4.12f} = {:4.12f} | {}'.format(
                #     epoch + 1, pair[1], l5, error, pair
                # )
                # Verify if the network has converged
                if abs(error) <= tol:
                    print '[CONVERGED]'
                    break
                # Initialize the backpropagation algorithm
                self.backward_pass(pair[0], error, 0.1)
                epoch += 1
            print '[ONLINE] {:4} - {:4.8f} - {:4.8f} = {:4.8f}'.format(
                epoch, pair[1], l5, error
            )

        p = 1
        for pair in data:
            l1, l2, l3, l5 = self.forward_pass(pair[0], pair[1], False)
            print 'Pair {:3} -> {:4.5f} --- {:4.12f}'.format(p, pair[1], l5)
            p += 1

    def forward_pass(self, entries, expected, prod=False, newrow=False):
        """ This method will compute the outputs from layers 1 to 4. The fourth
        layer will be calculated after finding the parameters with a Least
        Square Estimation.

        Parameters
        ----------
        entries : list of double
            The inputs to the ANFIS
        expected : double
            The expected value to the given parameters
        prod : boolean
            If set to True the min operation will be used to compute the layer
            2 outputs. Otherwise, the product operation will be used. Defaults
            to False.
        newrow : boolean
            Inform that this epoch is training a new pair, and this should be
            added to the linear system. Defaults to false.

        Returns
        ------
        layer_1 : list of double
            The outputs from layer one
        layer_2 : list of double
            The outputs from layer two
        layer_3 : list of double
            The outputs from layer three
        """
        layer_1 = []
        last_idx = 0
        for entry, subset, idx in zip(entries, self.subsets, self._sets):
            layer_1.append(subset.evaluate(entry, self._prec[last_idx:idx]))
            last_idx = idx

        layer_2 = np.empty([])
        if prod:
            layer_2 = self._product_operation(layer_1)
        else:
            layer_2 = self._min_operation(layer_1)

        layer_3 = layer_2 * (1.0 / layer_2.sum())

        consequents = self._find_consequents(
            layer_2, layer_3, expected, newrow
        )
        # Multiply, element-wise the consequent membership by the layer_3
        layer_4 = self.coef_matrix * consequents
        # pdb.set_trace()
        layer_5 = layer_4.sum()
        return layer_1, layer_2, layer_3, layer_5

    def backward_pass(self, entries, error, k=0.01):
        """ Computes the derivative of each membership function in the first
        layer and update the precedent parameters.

        Parameters
        ----------
        entries : list of double
            The network inputs
        error : double
            The error for the current epoch
        k : double
            A double to be used on the learning rate. Defaults to 0.1
        """
        err = -2 * error
        derivs = []
        last_idx = 0
        # Compute the derivative for each membership function, and its params
        for i in xrange(len(self._sets)):
            idx = self._sets[i]
            derivs.extend(
                self.subsets[i].derivs_at(entries[i], self._prec[last_idx:idx])
            )
            last_idx = idx
        derivs = np.array(derivs)
        # Sum of derivatives
        derivs_sqrsum = 0
        for rs in derivs:
            derivs_sqrsum += sum([x ** 2 for x in rs])

        eta = k
        if derivs_sqrsum != 0:
            eta = k / sqrt(derivs_sqrsum)

        # Update the precedent parameters by delta
        self._prec = self._prec + (derivs * (-eta * err))
