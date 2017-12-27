from random import random
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

    def __init__(self, pre, size, rules, consequents, inference):
        """ This method initialize a new instance of an ANFIS.
        Keyword arguments:
        pre -- The precedent fuzzy subsets of the Inference System
        size -- Number of neurons in the second and third layers
        rules -- The IF-THEN rules for the inference system
        consequents -- The consequent parameters set fo the inference system
        inference -- The inference strategy of the system
        """
        self.precedents = pre
        self.consequents = consequents
        self.inference = inference
        self.size = size if size > 0 else Anfis.MIN_SIZE
        self.map12 = []
        self.w23 = [size][size]
        self.w34 = [size][size]
        self.w45 = [len(rules)]
        self.eta = 0.002
        self.numOfLabels = 3

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
                errorList.add(diffParams)
                break
            elif len(conSet) != self.numOfLabels + 1:
                errorList.add(nonLinParams)

        if len(errorList) > 0:
            for error in errorList:
                print error
            exit()

    def fowardPass(self, inputs):
        """ This will feed the network with the given inputs, that is, will
        run one epoch with the inputs.

        Keyword arguments:
        inputs -- A feature vector to feed the network
        """
        x = 10
        precOutput = []

        # Layer 1
        for precedent in self.precedents:
            precOutput.append(precedent.membershipValue(x))

        # Layer 2 input -> output
        layerTwo = [1 for i in range(self.numOfLabels)]
        for i in range(self.precedents):
            node = (i + 1) % self.numOfLabels
            layerTwo[node] *= precOutput[i]

        # Layer 3 input -> output
        layerTwoSummation = sum(layerTwo)
        layerThree = [0 for i in range(self.numOfLabels)]
        for node in range(self.numOfLabels):
            layerThree[node] = layerTwo[node] / layerTwoSummation

        # Layer 4 input -> output
        layerFour = [0 for i in range(self.numOfLabels)]
        linFunc = 1
        for node in range(self.numOfLabels):
            for param in range(self.numOfLabels - 1):
                linFunc += param * inputs[param]
            linFunc += self.consequents[node][self.numOfLabels]
            layerFour[node] = layerTwo[node] + layerThree[node] * linFunc

        # Layer 5
        result = self.inference(layerFour, layerThree)
        print result
