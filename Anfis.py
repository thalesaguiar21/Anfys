from __future__ import print_function
from Errors import log_print
from SpeechUtils import lse
from numpy import array
from math import sqrt
from itertools import product


class Anfis():
    """ This class represents a Type-1 Adaptive Neural Fuzzy Inference
    System (ANFIS). This structure was suggested by Jang (1993). The
    Type-1 ANFIS is a five-layer feed foward neural network. In this work, the
    purpose is to apply this structure in the Speech Reconigtion problem. This
    Inference system follows the basic IF-THEN rule system from Takagi and
    Sugeno.
    """
    __MIN_SIZE = 2

    def __init__(self, pre, consequents, cons_params):
        """ This method initialize a new instance of an ANFIS.

        Parameters
        ----------
        pre : list of FuzzySubset
            The first layer, or precedent membership functions
        consequents : list of LinguisticLabel
            The fourth layer, or the consequent membership functions
        cons_params : 2D list of double
            The parameters p and q of each consequent memebership function
        """
        self.__numOfLabels = len(pre[0].labels)
        self.__numOfRules = len(pre) ** self.__numOfLabels
        self.__rules = []
        self.cons_params = [[0] * 2 for i in range(self.__numOfRules)]
        self.precedents = pre
        self.consequents = consequents
        self.__create_rules()

    def __create_rules(self):
        label_set = [i for i in range(self.__numOfLabels)]
        labels_sets = [label_set for i in range(len(self.precedents))]
        self.__rules = product(*labels_sets)

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
        layerTwo : list of double
            The outputs from the second layer, using product
        layerFour : list of double
            The output of the layer four, that is, each element is an the crisp
            value of a consequent membership function multiplied by the
            normalized value of each rule's firing strength.
        """
        print('Initiating forward pass...')
        precOutput = []

        # Layer 1
        for (fuzz, entry) in zip(self.precedents, inputs):
            precOutput.append(fuzz.evaluate(entry))
        log_print('Layer 1 output')
        log_print('-' * len('Layer 1 output'))
        log_print(array(precOutput))
        log_print('=' * 100)

        # Layer 2 input -> output
        layerTwo = [1.0 for i in range(self.__numOfLabels)]
        for i in range(self.__numOfLabels):
            for prec in precOutput:
                layerTwo[i] *= prec[i]
        log_print('Layer 2 output')
        log_print('-' * len('Layer 1 output'))
        log_print(array(layerTwo))
        log_print('=' * 100)

        # Layer 3 input -> output
        l2_Sum = sum(layerTwo)
        layerThree = [float(x) / l2_Sum for x in layerTwo]
        log_print('Layer 3 output')
        log_print('-' * len('Layer 1 output'))
        log_print('Sum of layer 2 = ' + str(l2_Sum) + '\n')
        log_print(array(layerThree))
        log_print('=' * 100)

        # print('Before update:\n' + str(array(self.cons_params)))

        self.update_consequents(layerTwo, layerThree, expected)

        # Layer 4 input -> output
        layerFour = []
        for i in range(self.__numOfLabels):
            fi = self.consequents[i].membership_degree(
                layerTwo[i],
                self.cons_params[i]
            )
            layerFour.append(fi * layerThree[i])
        log_print('\nLayer 4 output')
        log_print('-' * len('Layer 1 output'))
        log_print(array(layerFour))
        log_print('=' * 100)
        print('End of forward pass!')
        return layerTwo, layerFour

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
        for i in range(self.__numOfRules):
            ki = (layer3[i] * layer2[i])
            coef_line.append(ki)
            coef_line.append(-ki)
        coefMatrix = []
        coefMatrix.append(coef_line)

        # pdb.set_trace()
        print('Coefficient matrix dim: ' + str(len(coefMatrix)))

        self.cons_params = lse(coefMatrix, [sum(expected)], lamb=lamb)
        # pdb.set_trace()
        self.cons_params = [
            self.cons_params[v, 0] for v in range(2 * self.__numOfRules)
        ]

        total = 0
        for i in range(len(self.cons_params)):
            total += self.cons_params[i] * coefMatrix[0][i]
        print('LSE approximation result: ' + str(total))

        tmp_params = []
        for i in range(0, 2 * self.__numOfRules, 2):
            tmp_params.append(
                [self.cons_params[i], self.cons_params[i + 1]]
            )
        self.cons_params = tmp_params

    def backward_pass(self, expected, inputs, l4Input, layerFour):
        """ Implements the backward pass of the neural network. In this case,
        the backward pass for an ANFIS consider that the consequent parameters
        are optimal, and therefore they are fixed in this stage.

        Parameters
        ----------
        expected : list of double
            The expected values for the output layer
        inputs : list of double
            The input feeded to neural network
        layerFour : list of double
            The output vector of layer four
        target : list of double
            A target output vector

        Returns
        -------
        error : double
            A double value of the data loss (error) for this epoch
        """
        print('Initializing backward pass...')

        dE_dO = []
        for i in range(self.__numOfLabels):
            dE_dO.append(-2 * (expected[i] - layerFour[i]))
        # print('dE_dO: \n' + str(array(dE_dO)))

        dO_dW = []
        for (consequent, entry) in zip(self.consequents, l4Input):
            dO_dW.append(consequent.derivative_at(entry, 'lamb', []))
        # print('dO_dW: \n' + str(array(dE_dO)))

        # Computing the derivative of each label
        dW_dAlpha = []
        for (precFuzzySet, inp) in zip(self.precedents, inputs):
            dW_dAlpha.append(precFuzzySet.derivs_at(inp))
        dW_dAlpha = array(dW_dAlpha)
        # print('dW_dAlpha: \n' + str(array(dE_dO)))

        dE_dAlpha = []
        for fuzzDerivs in dW_dAlpha:
            tmp = []
            for i in range(self.__numOfLabels):
                tmp.append([
                    dE_dO[i] * dO_dW[i] * fuzzDerivs[i][k] for k in range(3)
                ])
            dE_dAlpha.append(tmp)

        grad_sum = 0
        for fuzzDerivs in dE_dAlpha:
            grad_sum += sum(sum(array(fuzzDerivs)))

        k = 0.1
        eta = k / sqrt(grad_sum ** 2)

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
            print('\n\nTraining for data \nINPUT:\n{}\nOUTPUT:\n{}\n'.format(
                array(feature), array(expected)
            ))
            print('\nEXPECTED PREDICTION: ' + str(sum(expected)))
            print('=' * 150)
            epoch = 0
            converged = False
            while epoch < nEpochs and not converged:
                l2, l4 = self.forward_pass(feature, expected)
                error = 0
                for (target, output) in zip(expected, l4):
                    error += target - output
                error = error ** 2
                print(
                    'Error for {}-th epoch is {}\n'.format(epoch + 1, error),
                    end=''
                )
                converged = error <= errTolerance
                # if not converged:
                #     self.backward_pass(expected, feature, l2, l4)
                epoch += 1
                errors.append(error)
                print()
            l2, l4 = self.forward_pass(feature, expected)
            prediction = sum(l4)
            if converged:
                print('Convergence occurred at epoch {}'.format(epoch))
            print('Final predition was {} with {} of error!'.format(
                prediction, errors[-1]))
