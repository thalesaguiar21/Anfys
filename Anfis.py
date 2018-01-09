from __future__ import print_function
from Errors import err, log_print
from exceptions import AttributeError, IndexError
from SpeechUtils import lse, index
from numpy import array
from random import random


class Anfis():
    """ This class represents a Type-1 Adaptive Neural Fuzzy Inference
    System (ANFIS). This structure was suggested by Jang (1993). The
    Type-1 ANFIS is a five-layer feed foward neural network. In this work, the
    purpose is to apply this structure in the Speech Reconigtion problem. This
    Inference system follows the basic IF-THEN rule system from Takagi and
    Sugeno.
    """
    MIN_SIZE = 2

    def __init__(self, pre, consequents, inference):
        """ This method initialize a new instance of an ANFIS.

        Parameters
        ----------
        pre : list of FuzzySubset
            The first layer, or precedent membership functions
        consequents : list of LinguisticLabel
            The fourth layer, or the consequent membership functions
        inference : InferenceStrategy
            The inference strategy, or the fifth layer
        """
        self.consParams = []
        self.precedents = pre
        self.consequents = consequents
        self.inference = inference
        self.eta = 0.002
        self.numOfLabels = len(pre[0].labels)
        self.steps = [random()]
        for i in range(1, self.numOfLabels):
            self.steps[i] = self.steps[i - 1] + random()

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
            self.numOfLabels = numOfLabels

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
        infered : double
            A duffuzified value representing the inference strategy
            result.
        outputVector : list of double
            A list with the output from the fourth layer
        """
        log_print('Initiating forward pass...')
        precOutput = []

        # Layer 1
        j = 0   # Input index
        for fuzz in self.precedents:
            precOutput.append(fuzz.evaluate(inputs[j]))
            j += 1
        log_print('Layer 1 output')
        log_print('-' * len('Layer 1 output'))
        log_print(array(precOutput))
        log_print('=' * 100)

        # Layer 2 input -> output
        layerTwo = [1.0 for i in range(self.numOfLabels)]
        for i in range(self.numOfLabels):
            for prec in precOutput:
                layerTwo[i] *= prec[i]
        log_print('Layer 2 output')
        log_print('-' * len('Layer 1 output'))
        log_print(layerTwo)
        log_print('=' * 100)

        # Layer 3 input -> output
        l2_Sum = sum(layerTwo)
        layerThree = [float(x) / l2_Sum for x in layerTwo]
        log_print('Layer 3 output')
        log_print('-' * len('Layer 1 output'))
        log_print(layerThree)
        log_print('=' * 100)

        # Layer 4 input -> output
        layerFour = []
        for i in range(self.numOfLabels):
            fi = self.consequents[i].membership_degree(
                layerTwo[i],
                self.consParams[i]
            )
            layerFour.append(fi * layerThree[i])
        log_print('Layer 4 output')
        log_print('-' * len('Layer 1 output'))
        log_print(layerFour)
        log_print('=' * 100)

        log_print('Updating consequent parameters...')
        coefMatrix = [[layerTwo[i]] * 2 for i in range(len(self.consParams))]
        log_print('Coefficient matrix is...\n{}\n'.format(array(coefMatrix)))

        rsMatrix = [[0] * len(self.consParams) for i in range(len(layerFour))]
        for i in range(len(layerFour)):
            if i == index(expected):
                rsMatrix[i][i] = layerFour[i] + 0.002
            else:
                rsMatrix[i][i] = layerFour[i] - 0.002

        log_print('Result matrix is\n{}'.format(array(rsMatrix)))
        log_print('Running LSE...', end_='')

        self.consParams = lse(self.consParams, rsMatrix, lamb=0.03)
        self.consParams = self.consParams.tolist()
        self.consParams = [[x, y] for (x, y) in zip(*self.consParams)]
        log_print(
            'Done!\nNew parameters are...\n{}'.format(array(self.consParams))
        )

        # Layer 5
        result = self.inference.infer(layerFour)
        log_print('Done!')
        return result, layerFour

    def backward_pass(self, layerFour, target, threshold):
        """ This method is a application of the Hybrid Learning Algorithm
        proposed by Jang (1993) for an Adaptive Neural Fuzzy Inference System.
        For each Backward pass, the consequent parameters are updated by one
        run os Least Square Estimation, while the premise parameters are
        updated by Backpropagation.

        Parameters
        ----------
        layerFour : list of double
            The output vector of layer four
        threshold : double
            The error tolerance

        Returns
        -------

        Raises
        ------
        """
        log_print('Initializing backward pass...')
        error = random()
        convergence = error <= threshold
        if convergence is True:
            log_print('The system has converged!')
        else:
            pass

        log_print('Error is ' + str(error))
        log_print('Finished an epoch!')
