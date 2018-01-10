from __future__ import print_function
from Errors import err, log_print
from exceptions import AttributeError, IndexError
from SpeechUtils import lse, index, step
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

        self.update_consequents(layerTwo, layerFour, expected, 2e-5)

        # Layer 5
        result = self.inference.infer(layerFour)
        log_print('Done!')
        log_print('Inferred phoneme is {}, target was {}'.format(
            step(result),
            expected)
        )
        return result, layerFour

    def update_consequents(self, l2Output, l4Output, expected, zeta):
        """ Update the consequent parameters with a Least Square Estimation

        Parameters
        ----------
        l2Output : list of double
            The output from the second layer
        l4Output : list of double
            The output
        expected : str
            The expected phoneme
        zeta : double
            A small double value to create a desired output. This will be used
            to update the outputs.
        """
        log_print('Updating consequent parameters...')
        coefMatrix = [[l2Output[i]] * 2 for i in range(len(self.consParams))]
        log_print('Coefficient matrix is...\n{}\n'.format(array(coefMatrix)))

        rsMatrix = [[0] * len(self.consParams) for i in range(len(l4Output))]
        for i in range(len(l4Output)):
            if i == index(expected):
                rsMatrix[i][i] = l4Output[i] + zeta
            else:
                rsMatrix[i][i] = l4Output[i] - zeta

        log_print('Result matrix is\n{}'.format(array(rsMatrix)))
        log_print('Running LSE...', end_='')

        self.consParams = lse(self.consParams, rsMatrix, lamb=0.03)
        self.consParams = self.consParams.tolist()
        self.consParams = [[x, y] for (x, y) in zip(*self.consParams)]
        log_print(
            'Done!\nNew parameters are...\n{}'.format(array(self.consParams))
        )

    def backward_pass(self, inputs, layerFour, target):
        """ Implements the backward pass of the neural network. In this case,
        the backward pass for an ANFIS consider that the consequent parameters
        are optimal, and therefore they are fixed in this stage.

        Parameters
        ----------
        inputs : list of double
            The input feeded to neural network
        layerFour : list of double
            The output vector of layer four
        target : list of double
            A target output vector

        Returns
        -------

        Raises
        ------
        """
        log_print('Initializing backward pass...')
        error = random()

        log_print('Error is ' + str(error))
        log_print('Finished an epoch!')
