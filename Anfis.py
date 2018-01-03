# from random import random
from sys import exit


class Anfis():
    """ This class represents a Type-1 Adaptive Neural Fuzzy Inference
    System (ANFIS). This structure was suggested by Jang (1993). The
    Type-1 ANFIS is a five-layer feed foward neural network. In this work, the
    purpose is to apply this structure in the Speech Reconigtion problem. This
    Inference system follows the basic IF-THEN rule system from Takagi and
    Sugeno.
    """
    MIN_SIZE = 2

    def __init__(self, pre, numOfLabels, consequents, inference, eta):
        """ This method initialize a new instance of an ANFIS.

        Keyword arguments:
        pre -- The precedent fuzzy subsets of the Inference System
        numOfLabels -- Number of neurons in the second and third layers
        consequents -- The consequent parameters set fo the inference system
        inference -- The inference strategy of the system
        eta -- Learning rate
        """
        self.precedents = pre
        self.consequents = consequents
        print consequents
        self.inference = inference
        self.eta = eta
        self.numOfLabels = numOfLabels

    def validate(self, inputs):
        """ Verify the ANFIS before the foward pass.

        Keyword arguments:
        inputs -- the data to feed in the network
        """
        errorList = []
        diffParams = '''[ERROR] The number of consequent parameters must
        be homogeneous'''
        nonLinParams = '''[ERROR] The number of consequent parameters must be
        equal to the number of linguistic labels plus 1'''
        numOfParams = len(self.consequents[0])
        for conSet in self.consequents:
            if len(conSet) != numOfParams:
                errorList.append(diffParams)
                break
            elif len(conSet) != self.numOfLabels + 1:
                errorList.append(nonLinParams)

        if len(errorList) > 0:
            for error in errorList:
                print error
            exit()

    def fowardPass(self, inputs):
        """ This will feed the network with the given inputs, that is, will
        run one epoch with the inputs.

        Keyword arguments:
        inputs -- A feature vector to feed the network

        Return:
        double -- A duffuzified value representing the inference strategy
        result
        """
        print 'Initiating forward pass...'
        precOutput = []

        # Layer 1
        j = 0   # Input index
        for fuzz in self.precedents:
            for node in fuzz:
                precOutput.append(node.membershipValue(inputs[j]))
            j += 1

        # Layer 2 input -> output
        layerTwo = [1.0 for i in range(self.numOfLabels)]
        for i in range(len(precOutput)):
            node = i % self.numOfLabels
            layerTwo[node] *= precOutput[i]

        # Layer 3 input -> output
        layerTwoSummation = sum(layerTwo)
        layerThree = [0 for i in range(self.numOfLabels)]
        for node in range(self.numOfLabels):
            layerThree[node] = layerTwo[node] / layerTwoSummation

        # Layer 4 input -> output
        layerFour = [0 for i in range(self.numOfLabels)]
        linFunc = 0
        for node in range(self.numOfLabels):
            for param in range(self.numOfLabels):
                linFunc += self.consequents[node][param] * inputs[param]
            linFunc += self.consequents[node][self.numOfLabels]
            layerFour[node] = layerTwo[node] + layerThree[node] * linFunc
            linFunc = 0

        # Layer 5
        result = self.inference.infer(layerFour)
        return result

    def backwardPass(self, error, inputs):
        """ This method is a application of the Hybrid Learning Algorithm
        proposed by Jang (1993) for an Adaptive Neural Fuzzy Inference System.

        Keyword arguments:
        error -- The error obtained from the forward pass
        inputs -- The set of input variables
        """

        pass
