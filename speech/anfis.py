from __future__ import division
from __future__ import print_function
from itertools import product
from fuzzy.subsets import FuzzySet
from data import file_helper as fhelper
from speech import utils as sputils
from fuzzy import mem_funcs as mfs
from math import sqrt, isinf
from random import randint
import pdb

# import pdb
from time import clock
import numpy as np
import sys
sys.path.append('../')


class BaseModel(object):

    def __init__(self, mf_n, inp_n, mem_func=mfs.BellTwo()):
        self.mf_n = mf_n  # Number of MFs in each set
        self.inp_n = inp_n  # Number of inputs
        self.rules_n = mf_n ** inp_n  # Number of network rules
        self._prec = self.init_prec_params(mem_func)  # Initial prec params
        self._rule_set = self._create_rules()
        self.subsets = [FuzzySet(mem_func) for i in xrange(self.inp_n)]
        self._errors = []

    def init_prec_params(self, mf):
        param_n = len(mf.parameters)
        param_t = self.mf_n * self.inp_n
        if param_n == 2:
            return np.array(self.init_twoparam_prec(param_t))
        elif param_n == 3:
            return np.array(self.init_threeparam_prec(param_t))

    def init_twoparam_prec(self, size):
        return [[randint(1, 10), randint(1, 10)] for _ in range(size)]

    def init_threeparam_prec(self, size):
        return [
            [randint(1, 10), randint(1, 10), randint(1, 10)]
            for _ in range(size)
        ]

    def _create_rules(self):
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
        if self.mf_n is not None and self.mf_n > 0:
            rules_set = [range(self.mf_n) for i in xrange(self.inp_n)]
            return np.array([comb for comb in product(*rules_set)])
        return None

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

    def learn_hybrid_online(self, data, threshold=1e-10, max_epochs=500):
        raise NotImplementedError()

    def _find_consequents(values, weights, new_line=False):
        raise NotImplementedError()

    def forward_pass(self, entries, expected, min_prod=False, new_row=False):
        raise NotImplementedError()

    def backward_pass(self, layer_1, layer_2, layer_5, error):
        raise NotImplementedError()


class TsukamotoModel(BaseModel):
    """ Class to represent an Artifical Neural Fuzzy Inference System which
    uses the Tsukamoto Fuzzy Inference System.
    """

    def __init__(self, mf_n, inp_n, cons_fun, mem_func=mfs.BellTwo()):
        """ Initialize a new Tsukamoto Fuzzy Inference System

        Parameters
        ----------
        mf_n : int
            The size of each fuzzy set in the precendents layer
        inp_n : int
            Number of inputs to the network
        cons_fun : MembershipFunction
            A piecewise linear approximation of a monotonic function
        mem_func : MembershipFunction
            A MembershipFcuntion to be used on the precedent layer
        """
        super(TsukamotoModel, self).__init__(mf_n, inp_n, mem_func)
        self.cons_fun = cons_fun
        col = self.rules_n * len(cons_fun.parameters)
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
        newrow : boolean, defaults to False
            If set to true a new equation is added to the
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
        newrow : boolean, defaults to False
            Inform that this epoch is training a new pair, and that should be
            added to the linear system.
        """
        self._build_coefmatrix(values, weights, expec_result, newrow)
        result = sputils.lse_online(self.coef_matrix, self.expected)
        return [rs[0] for rs in result]

    def learn_hybrid_online(
            self, data, smp, tol=1e-6, max_epochs=250, tsk=None, heurist=True):
        """ Train the ANFIS with the given data pairs.

        Parameters
        ----------
        data : list of pars of list and integer
            The training data set. For instance [([2, 3, 4], 10)], where 10 is
            the expected value for [2, 3, 4] entry.
        tol : double, defaults to 10%
            The error tolerance.
        max_epochs : integer, defaults to 500
            The maximum number of epochs for each pair.
        tsk : string, defaults to None
            The task name for the inputs
        """
        for pair in data:
            if len(pair[0]) != self.inp_n:
                raise ValueError('Number of inputs must match number of sets!')

        qtd_data = len(data)
        p = 1
        with open('results/' + tsk, 'a') as _file:
            for pair in data:
                sputils.p_progress(qtd_data, p, tsk[:-1] + ' -> ' + str(smp))
                errs = []
                addsub_k = [0, 0]
                ttime = 0
                k = 0.1
                phn = 0
                start = clock()  # Starting time of an epoch
                for epoch in xrange(1, max_epochs + 1):
                    newrow = epoch == 0
                    layers_out = self.forward_pass(pair[0], pair[1], newrow)
                    # Update K from the 3th epoch
                    if epoch > 2 and heurist:
                        k, addsub_k = update_k(k, addsub_k, errs)
                        del errs[0]

                    errs.append((pair[1] - layers_out[4]) ** 2)  # Layer 5
                    phn = layers_out[4]
                    # Verify if the network has converged
                    if errs[-1] <= tol:
                        break
                    # Initialize the backpropagation algorithm
                    self.backward_pass(pair[0], errs[-1], layers_out, k)
                # Compute elapsed time and save to file
                p += 1
                ttime += clock() - start
                fhelper.w(_file, epoch, k, errs[-1], ttime, phn)

    def forward_pass(self, entries, expected, newrow=False):
        """ This method will compute the outputs from layers 1 to 4. The fourth
        layer will be calculated after finding the parameters with a Least
        Square Estimation.

        Parameters
        ----------
        entries : list of double
            The inputs to the ANFIS
        expected : double
            The expected value to the given parameters
        newrow : boolean, defaults to False
            Inform that this epoch is training a new pair, and this should be
            added to the linear system.

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
        idx = 0
        for entry, subset in zip(entries, self.subsets):
            init = idx * self.mf_n
            layer_1.append(
                subset.evaluate(entry, self._prec[init:init + self.mf_n])
            )
            idx += 1
        layer_2 = self._product_operation(layer_1)
        # layer_2 is an numpy array
        layer_3 = layer_2 * (1.0 / layer_2.sum())
        consequents = self._find_consequents(
            layer_2, layer_3, expected, newrow
        )
        # Compute the membership value of each consequent function
        layer_4 = np.empty(self.rules_n)
        for i in xrange(self.rules_n):
            k = i * 2
            fi = self.cons_fun.membership_degree(
                layer_2[i], *consequents[k: k + 2]
            )
            layer_4[i] = fi * layer_3[i]
        layer_5 = layer_4.sum()
        return layer_1, layer_2, layer_3, layer_4, layer_5

    def backward_pass(self, entries, error, layers_out, k=0.1):
        """ Computes the derivative of each membership function in the first
        layer and update the precedent parameters.

        Parameters
        ----------
        entries : list of double
            The network inputs
        error : double
            The error for the current epoch
        k : double, defaults to 0.1
            A double to be used on the learning rate.
        layers_out : matrix
            A matrix with the outputs from each layer in forward pass
        """
        # Unpack the output of each layer
        de_do = self.compute_de_do(*layers_out[:-1])
        # Derivative of MFs for each alpha dO_dAlpha
        idx = 0
        derivs = []
        for i in xrange(self.inp_n):
            params = self._prec[i * self.mf_n:i * self.mf_n + self.mf_n]
            derivs.extend(self.subsets[i].derivs_at(entries[i], params))
            idx += 1
        derivs = np.array(derivs)

        de_dalpha = derivs * (-2 * sqrt(error)) * de_do
        denom = sqrt((de_dalpha ** 2).sum())
        if denom == 0:
            eta = 0.001
        else:
            eta = k / denom
        if isinf(eta):
            eta = 0.001
        # Update the precedent parameters
        delta_alpha = -eta * de_dalpha
        self._prec = self._prec + delta_alpha

    def compute_de_do(self, l1, l2, l3, l4):
        """ Calculate dE_dO for this epoch

        Parameters
        ----------
        l1, l2, l3, l4: list of double
            The respective layer output

        Returns
        -------
        Derivative of E with respect to O*
        """
        do5_do4, do4_do3, do3_do2, do2_do1 = [0 for _ in range(4)]
        l2_sum = l2.sum()
        for i in xrange(self.rules_n):
            # Compute do5_do4
            do5_do4 += 1.0
            # Compute do4_do3
            do4_do3 += l4[i] / l3[i]
            # Compute do3_do2
            do3_do2 += (l2_sum - l2[i]) ** 2
            for (prec, mf_set) in zip(self._rule_set[i], range(self.inp_n)):
                do2_do1 += l2[i] / l1[mf_set][prec]
        return do5_do4 * do4_do3 * do3_do2 * do2_do1


def update_k(k, addsub, errors):
    """ Update the value of k.
    1 -> If theres four consecutive error increases,
         then k is increased by 10%.
    2 -> If theres four consecutive error decreases,
         then k is increased by 10%.

    Parameters
    ----------
    k : double
        The value to be updated
    addsub : vector of int with size 2
        The increase and decrease counters
    errors : list of double
        The list of errors

    Returns
    -------
    The updated value of 'k' and 'addsub'.
    """
    if errors[-1] <= errors[-2]:
        addsub[0] += 1
        addsub[1] = 0
    elif errors[-1] > errors[-2]:
        addsub[1] += 1
        addsub[0] = 0
    if addsub[0] == 4:
        k *= 1.10
        addsub[0] = 0
    if addsub[1] == 4:
        k *= 0.9
        addsub[1] = 0
    return k, addsub
