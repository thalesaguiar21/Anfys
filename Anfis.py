from __future__ import print_function, division
from numpy import array
from math import sqrt
from itertools import product
from speech.interface import *
from speech.utils import lse_online
from errors.debug import Debugger


class Anfis():
    """ This class represents a Type-1 Adaptive Neural Fuzzy Inference
    System (ANFIS). This structure was suggested by Jang (1993). The
    Type-1 ANFIS is a five-layer feed foward neural network. In this work, the
    purpose is to apply this structure in the Speech Reconigtion problem. This
    Inference system follows the basic IF-THEN rule system from Tsukamoto FIS.
    """

    def __init__(self, pre, consequents):
        """ This method initialize a new instance of an ANFIS.

        Parameters
        ----------
        pre : list of FuzzySubset
            The first layer, or precedent membership functions
        consequents : list of LinguisticLabel
            The fourth layer, or the consequent membership functions
        """
        self.__debugger = Debugger(False)
        self.__num_of_labels = pre[0]._num_of_labels
        self.__num_of_rules = self.__num_of_labels ** len(pre)
        self.__rules = []
        self.cons_params = [[0] * 2 for i in range(self.__num_of_rules)]
        self.precedents = pre
        self.consequents = consequents
        self.__create_rules()

    def __create_rules(self):
        """ Compute the cartesian product of the precedents and initialize
        the __rules attribute.
        """
        label_set = range(self.__num_of_labels)
        labels_sets = [label_set for i in range(len(self.precedents))]
        self.__rules = [rule for rule in product(*labels_sets)]

    def __min_output(self, prec_output):
        """ Computes the output of the Layer Two using the min of the inputs

        Parameters
        ----------
        prec_output : list of double
            The output from the previous layer
        """
        outputs = []
        for rule in self.__rules:
            minimum = prec_output[0][rule[0]]
            for (mem_outs, label) in zip(range(len(prec_output)), rule):
                if minimum > prec_output[mem_outs][label]:
                    minimum = prec_output[mem_outs][label]
            outputs.append(minimum)
        return outputs

    def __product_output(self, prec_output):
        """ Computes the output of the Layer Two using the product of the
        inputs.

        Parameters
        ----------
        prec_output : list of double
            The output from the previous layer
        """
        outputs = []
        for rule in self.__rules:
            prod = 1.0
            for (mem_outs, label) in zip(range(len(prec_output)), rule):
                prod *= prec_output[mem_outs][label]
            outputs.append(prod)
        return outputs

    def forward_pass(self, inputs, expected):
        """ This will feed the network with the given inputs, that is, will
        feed the network until layer 5

        Parameters
        ----------
        inputs : list of double
            A feature vector to feed the network
        expected : str
            The string correponding to the phoneme

        Returns
        -------
        layer_two : list of double
            The outputs from the second layer, using product
        layer_four : list of double
            The output of the layer four, that is, each element is an the crisp
            value of a consequent membership function multiplied by the
            normalized value of each rule's firing strength.
        """
        print('Initiating forward pass...')

        # Layer 1
        prec_output = []
        for (fuzz, entry) in zip(self.precedents, inputs):
            prec_output.append(fuzz.evaluate(entry))
        self.__debugger.print(layer_output_msg(prec_output, 1))

        # Layer 2 input -> output
        layer_two = self.__min_output(prec_output)
        self.__debugger.print(layer_output_msg(layer_two, 2))

        # Layer 3 input -> output
        l2_Sum = sum(layer_two)
        layer_three = [float(x) / l2_Sum for x in layer_two]
        self.__debugger.print(layer_output_msg(layer_three, 3))

        self.update_consequents(layer_two, layer_three, expected, 0.95)

        # Layer 4 input -> output
        layer_four = []
        for i in range(self.__num_of_rules):
            fi = self.consequents[i].membership_degree(
                layer_two[i],
                *self.cons_params[i]
            )
            layer_four.append(fi * layer_three[i])
        self.__debugger.print(layer_output_msg(layer_four, 4))
        print('End of forward pass!')
        return layer_two, layer_four

    def update_consequents(self, layer2, layer3, expected, lamb=0.9):
        """ Update the consequent parameters with a Least Square Estimation

        Parameters
        ----------
        layer2 : list of double
            The output from the second layer
        layer3 : list of double
            The output from the third layer
        expected : list of double or double
            The expected value for the current input
        lamb : double
            THe forgetting factor, usually a small number between 0 and 1.
            Defaults to 0.9
        """
        print('Updating consequent parameters...')

        coef_line = []
        for i in range(self.__num_of_rules):
            ki = (layer3[i] * layer2[i])
            coef_line.append(ki)
            coef_line.append(-ki)
        coef_matrix = []
        coef_matrix.append(coef_line)

        self.cons_params = lse_online(
            coef_matrix,
            [[sum(expected)]],
            lamb,
            10 ** 10
        )

        self.cons_params = [
            self.cons_params[v, 0] for v in range(2 * self.__num_of_rules)
        ]

        total = 0
        for i in range(len(self.cons_params)):
            total += self.cons_params[i] * coef_matrix[0][i]
        print('Expected {} for LSE, got: {}'.format(sum(expected), total))

        tmp_params = []
        for i in range(0, 2 * self.__num_of_rules, 2):
            tmp_params.append(
                [self.cons_params[i], self.cons_params[i + 1]]
            )
        self.cons_params = tmp_params

    def backward_pass(self, expected, inputs, l4_input, l4_output, step=0.01):
        """ Implements the backward pass of the neural network. In this case,
        the backward pass for an ANFIS consider that the consequent parameters
        are optimal, and therefore they are fixed in this stage.

        Parameters
        ----------
        expected : list of double
            The expected values for the output layer
        inputs : list of double
            The input feeded to neural network
        l4_input : list of double
            The input vector for layer four
        step : double
            An small double value representing the step size of the gradient.
        """
        print('Initializing backward pass...')

        dE_dO = []
        for i in range(self.__num_of_rules):
            dE_dO.append(-2 * (expected[i] - l4_output[i]))

        dO_dW = []
        for i in range(self.__num_of_rules):
            dO_dW.append(self.consequents[i].derivative_at(
                l4_input[i], '', *self.cons_params[i])
            )

        # Computing the derivative of each label
        dW_dAlpha = []
        for (precFuzzySet, inp) in zip(self.precedents, inputs):
            dW_dAlpha.append(precFuzzySet.derivs_at(inp))
        dW_dAlpha = array(dW_dAlpha)

        dE_dAlpha = []
        for fuzzDerivs in dW_dAlpha:
            tmp = []
            for i in range(self.__num_of_labels):
                tmp.append([
                    dE_dO[i] * dO_dW[i] * fuzzDerivs[i][k] for k in range(3)
                ])
            dE_dAlpha.append(tmp)

        grad_sum = 0
        for fuzzDerivs in dE_dAlpha:
            grad_sum += sum(sum(array(fuzzDerivs)))

        # [REMEMBER] Remove the plus 1
        eta = step / (sqrt(grad_sum ** 2) + 1.0)

        for (precedent, deriv) in zip(self.precedents, dE_dAlpha):
            for i in range(len(deriv)):
                for j in range(len(deriv[i])):
                    precedent.params[i][j] -= eta * deriv[i][j]

        print('Finished backward pass!')

    def train_by_hybrid_online(self, nEpochs, errTolerance, trainingData):
        """ Train the ANFIS with the training data. Each (input, output) pair
        will run until a given number of epochs or the data loss (error)
        reaches a threshold.

        Parameters
        ----------
        nEpochs : int
            Maximum number of epochs for each training data
        errTolerance : double
            A small number for the desired error tolerance
        traningData : duple
        """
        errors = []
        for (feature, expected) in trainingData:
            self.__debugger.print(begin_training_msg(feature, expected))
            epoch = 0
            converged = False
            while epoch < nEpochs and not converged:
                l2, l4 = self.forward_pass(feature, expected)
                error = 0
                for (target, output) in zip(expected, l4):
                    error += target - output
                error = error ** 2
                errors.append(error)

                self.__debugger.print(epoch_error_msg(epoch + 1, error), False)

                converged = error <= errTolerance
                if not converged:
                    self.backward_pass(expected, feature, l2, l4)
                epoch += 1
                print()
            l2, l4 = self.forward_pass(feature, expected)
            prediction = sum(l4)
            if converged:
                convergence_msg(epoch + 1)
            print(end_epoch_msg(prediction, errors[-1]))
            print('Final output:\n{}'.format(array(l4)))

    def set_debug(self, debug):
        self.__debugger.debug = debug
