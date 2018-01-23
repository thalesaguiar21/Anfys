from __future__ import print_function
from Errors import err, log_print
from exceptions import AttributeError, IndexError
from SpeechUtils import lse
from numpy import array, zeros


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
        self.cons_params = cons_params
        self.precedents = pre
        self.consequents = consequents

    def validate(self, inputs):
        """ Verify the ANFIS before the foward pass.

        Parameters
        ----------
        inputs : list of double
            The data to feed in the network

        Returns
        -------
        valid : bool
            True if the ANFIS object satisfies the constraints, Flase otherwise

        Raises
        ------
        AttributeError
            If any of the given layers are None
        IndexError
            If the number of precedents is different of the number of inputs
        """
        if self.precedents is None:
            raise AttributeError(err['NO_PRECEDENTS'])
        if self.consequents is None:
            raise AttributeError(err['NO_CONSEQUENTS'])
        if self.inference is None:
            raise AttributeError(err['NO_INFERENCE'])
        elif len(inputs) != len(self.precedents):
            raise IndexError(err['INPUT_SIZE'])
        else:
            numOfLabels = len(self.precedents[0])
            for i in range(1, len(self.precedents)):
                if len(self.precedents[i]) != numOfLabels:
                    raise IndexError(err['DIFF_LABELS'])
            self.__numOfLabels = numOfLabels

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

    def update_consequents(self, layer2, layer3, expected, lamb=0.03):
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
            Defaults to 0.03
        """
        layerSize = 2 * len(self.cons_params)
        print('Updating consequent parameters...', end='')

        coefMatrix = []
        for i in range(layerSize):
            idx = i / 2
            if i % 2 == 0:
                coefMatrix.append(layer3[idx] * layer2[idx])
            else:
                coefMatrix.append(layer3[idx])

        self.cons_params = lse(coefMatrix, sum(expected), lamb=lamb)
        tmp_params = []
        for i in range(layerSize / 2):
            tmp_params.append(
                [self.cons_params[i, 0], self.cons_params[i + 1, 0]]
            )
        self.cons_params = tmp_params
        # self.cons_params = [[x, y] for (x, y) in zip(*self.cons_params)]
        print('Done!')
        # log_print(
        #    'Done!\nNew parameters are...\n{}'.format(array(self.cons_params))
        # )

    def backward_pass(self, err, inputs, layerFour, step_size=0.01):
        """ Implements the backward pass of the neural network. In this case,
        the backward pass for an ANFIS consider that the consequent parameters
        are optimal, and therefore they are fixed in this stage.

        Parameters
        ----------
        err : double
            The error from the fifth layer
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

        # Computing the derivative of each label
        derivatives = zeros(
            (self.__numOfLabels, len(self.precedents[0].params[0]))
        )
        fuzzSetDerivs = []
        for (precFuzzySet, inp) in zip(self.precedents, inputs):
            i = 0
            for label in precFuzzySet.labels:
                derivatives[i] = [
                    label.derivative_at(inp, v) for v in ['a', 'b', 'c']]
                i += 1
            fuzzSetDerivs.append(derivatives)
        fuzzSetDerivs = array(fuzzSetDerivs)

        for (prec, derivs) in zip(self.precedents, fuzzSetDerivs):
            prec.params = derivs

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
            print('=' * 150)
            epoch = 0
            converged = False
            while epoch < nEpochs and not converged:
                l2, l4 = self.forward_pass(feature, expected)
                errors = [(target - output) ** 2 for (target, output) in zip(
                    expected, l4)]
                error = sum(errors)
                # converged = error <= errTolerance
                # log_print(
                #     'Predicted {} and expected {}'.format(l5, expected)
                # )
                # if not converged:
                #     self.backward_pass(error, feature, l4, [])
                print(
                    'Error for {}-th epoch is {}\n'.format(epoch + 1, error)
                )
                epoch += 1
                errors.append(error)
            l2, l4 = self.forward_pass(feature, expected)
            prediction = sum(l4)
            if converged:
                print('Convergence occurred at epoch {}'.format(epoch))
            else:
                print('Final predition was {} with {} of error!'.format(
                    prediction, errors[-1]))
