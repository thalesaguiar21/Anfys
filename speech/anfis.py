from itertools import product
from fuzzy.subsets import FuzzySet
from data import file_helper as fhelper
from speech import utils as sputils
from fuzzy import mem_funcs as mfs
from math import sqrt, isinf
from random import randint
import pdb
from time import clock
from numpy import random, array, zeros
import sys
sys.path.append('../')


class BaseModel:

    def __init__(self, qtd_mf, qtd_inp, premise_type, consequent_type):
        self._qtd_mf = qtd_mf
        self._qtd_inp = qtd_inp
        self._qtd_rules = qtd_mf ** qtd_inp
        self._errors = []
        self._premise_type = premise_type
        self._consequent_type = consequent_type
        self._premises_params = None
        self._consequent_params = None

    def premises_per_mf(self):
        return self.premise_type.qtd_params()

    def premises_size(self):
        return self._qtd_mf * self._qtd_inp

    def consequent_per_mf(self):
        return self.consequent_type.qtd_params()

    def consequent_size(self):
        return self._qtd_rules


def configure(anfis):
    init_premise_paramters(anfis)
    create_rules(anfis)
    init_consequent_parameters(anfis)


def init_premise_paramters(anfis):
    anfis._premises_params = random.rand(
        anfis.premises_per_mf(), anfis.premises_size())


def create_rules(anfis):
    if anfis._qtd_mf is not None and anfis._qtd_mf > 0:
        rules_set = [range(anfis._qtd_mf) for i in range(anfis._qtd_inp)]
        return array([comb for comb in product(*rules_set)])
    return None


def init_consequent_parameters(anfis):
    anfis._consequent_params = zeros(
        anfis._qtd_rules, anfis.consequent_per_mf())


def train(anfis, data, max_epochs):
    for curr_epoch in max_epochs:
        for pair in data:
            forward_pass(anfis, pair)
            backward_pass(anfis)


def forward_pass(anfis, pair):
    entry, out = *pair


def backward_pass(anfis):
    pass




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
            self, data, smp, prompt, tol=1e-6, max_epochs=250, tsk=None,
            heurist=True):
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
        per_file = open('results/' + tsk.split('.')[0] + '.per', 'a')
        word = []
        with open('results/' + tsk, 'a') as _file:
            for pair in data:
                sputils.p_progress(qtd_data, p, tsk[:-1] + ' -> ' + str(smp))
                errs = []
                addsub_k = [0, 0]
                ttime = 0
                k = 0.1
                phn = 0
                start = clock()  # Starting time of an epoch
                for epoch in range(1, max_epochs + 1):
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
                word.append(phn)
                fhelper.w(_file, epoch, k, errs[-1], ttime, phn)
            # Compute and write PER
            pdb.set_trace()
            word = sputils.get_phn(tsk[:7], *word)

            if len(word) != len(data):
                raise ValueError('Word is bigger than prompt!')

            per = sputils.levenshtein_distance(prompt, word) / len(word)
            per_file.write('{:3}\t{:20.20f}\n'.format(smp, per))

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
        for i in range(self.rules_n):
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
        for i in range(self.inp_n):
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
        for i in range(self.rules_n):
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
