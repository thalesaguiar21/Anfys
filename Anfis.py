from __future__ import print_function
from Errors import err
from exceptions import AttributeError, IndexError


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
        pre : [FuzzySubset]
            The first layer, or precedent membership functions
        consequents : [LinguisticLabel]
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
        inputs : list
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

    def forwardPass(self, inputs):
        """ This will feed the network with the given inputs, that is, will
        feed the network until layer 5

        Parameters
        ----------
        inputs : list
            A feature vector to feed the network

        Returns
        -------
        infered : double
            A duffuzified value representing the inference strategy
            result.
        outputVector : [double]
            A list with the output from the fourth layer
        """
        print('Initiating forward pass...', end='')
        precOutput = []

        # Layer 1
        j = 0   # Input index
        for fuzz in self.precedents:
            precOutput.append(fuzz.evaluate(inputs[j]))

        # Layer 2 input -> output
        layerTwo = [1.0 for i in range(self.numOfLabels)]
        for i in range(self.numOfLabels):
            for prec in precOutput:
                layerTwo[i] *= prec[i]

        # Layer 3 input -> output
        layerTwoSummation = sum(layerTwo)
        layerThree = [0 for i in range(self.numOfLabels)]
        for node in range(self.numOfLabels):
            layerThree[node] = layerTwo[node] / layerTwoSummation

        # Layer 4 input -> output
        layerFour = []
        for i in range(self.numOfLabels):
            fi = self.consequents[i].membershipDegree(
                layerTwo[i],
                self.consParams[i]
            )
            layerFour.append(fi * layerThree[i])

        # Layer 5
        result = self.inference.infer(layerFour)
        print('Done!')
        return result, layerFour

    def backwardPass(self, error, inputs):
        """ This method is a application of the Hybrid Learning Algorithm
        proposed by Jang (1993) for an Adaptive Neural Fuzzy Inference System.
        """
        print('Initializing backward pass...', end='')
        print('Done!')
