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

    def __init__(self, pre, consequents, inference, eta):
        """ This method initialize a new instance of an ANFIS.

        Keyword arguments:
        inputParams -- The set of input parameters
        consParams -- The set of consequent Parameters
        pre -- The precedent fuzzy subsets of the Inference System
        consequents -- The consequent parameters set fo the inference system
        inference -- The inference strategy of the system
        eta -- Learning rate
        """
        self.consParams = []
        self.precedents = pre
        self.consequents = consequents
        self.inference = inference
        self.eta = eta
        self.numOfLabels = len(pre[0].labels)

    def validate(self, inputs):
        """ Verify the ANFIS before the foward pass.

        Keyword arguments:
        inputs -- the data to feed in the network
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
        run one epoch with the inputs.

        Keyword arguments:
        inputs -- A feature vector to feed the network

        Return:
        double -- A duffuzified value representing the inference strategy
        result
        """
        print('Initiating forward pass...')
        print('\n' * 2)
        precOutput = []
        times = 50

        # Layer 1
        print('Calculating layer one...')
        j = 0   # Input index
        for fuzz in self.precedents:
            precOutput.append(fuzz.evaluate(inputs[j]))
        print(precOutput)
        print('-' * times)

        # Layer 2 input -> output
        print('Calculating layer two...')
        layerTwo = [1.0 for i in range(self.numOfLabels)]
        for i in range(self.numOfLabels):
            for prec in precOutput:
                layerTwo[i] *= prec[i]
        print(layerTwo)
        print('-' * times)

        # Layer 3 input -> output
        print('Calculating layer three...')
        layerTwoSummation = sum(layerTwo)
        layerThree = [0 for i in range(self.numOfLabels)]
        for node in range(self.numOfLabels):
            layerThree[node] = layerTwo[node] / layerTwoSummation
        print(layerThree)
        print('-' * times)

        # Layer 4 input -> output
        print('Calculating layer four...')
        layerFour = []
        for i in range(self.numOfLabels):
            fi = self.consequents[i].membershipValue(
                layerTwo[i],
                self.consParams[i]
            )
            layerFour.append(fi * layerThree[i])
        print(layerFour)
        print('-' * times)

        # Layer 5
        print('Calculating layer 5...')
        result = self.inference.infer(layerFour)
        print(result)
        print('-' * 10)
        print('\n\nDone!')
        return result

    def backwardPass(self, error, inputs):
        """ This method is a application of the Hybrid Learning Algorithm
        proposed by Jang (1993) for an Adaptive Neural Fuzzy Inference System.

        Keyword arguments:
        error -- The error obtained from the forward pass
        inputs -- The set of input variables
        """
        print('Initializing backward pass...', end='')
        print('Done!')
