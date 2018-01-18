from Errors import err, log_print
from exceptions import AttributeError, IndexError
from SpeechUtils import lse
from numpy import array, zeros, eye, matrix, dot


class Anfis():
    """ This class represents a Type-1 Adaptive Neural Fuzzy Inference
    System (ANFIS). This structure was suggested by Jang (1993). The
    Type-1 ANFIS is a five-layer feed foward neural network. In this work, the
    purpose is to apply this structure in the Speech Reconigtion problem. This
    Inference system follows the basic IF-THEN rule system from Takagi and
    Sugeno.
    """
    MIN_SIZE = 2

    def __init__(self, pre, consequents, inference, alph):
        """ This method initialize a new instance of an ANFIS.

        Parameters
        ----------
        pre : list of FuzzySubset
            The first layer, or precedent membership functions
        consequents : list of LinguisticLabel
            The fourth layer, or the consequent membership functions
        inference : InferenceStrategy
            The inference strategy, or the fifth layer
        alph : MappedAlphabet
            The alphabet that will be used for phoneme recognition
        """
        self.__eta = 0.002
        self.consParams = []
        self.precedents = pre
        self.consequents = consequents
        self.inference = inference
        self.numOfLabels = len(pre[0].labels)
        self.alph = alph

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
        layerFour : list of double
            A duffuzified value representing the inference strategy
            result.
        outputVector : list of double
            A list with the output from the fourth layer
        layerTwo : list of double
            The outputs from the second layer
        layerThree : list of double
            Outputs from layer three
        """
        log_print('Initiating forward pass...', end_='')
        precOutput = []

        # Layer 1
        for (fuzz, entry) in zip(self.precedents, inputs):
            precOutput.append(fuzz.evaluate(entry))
        # log_print('Layer 1 output')
        # log_print('-' * len('Layer 1 output'))
        # log_print(array(precOutput))
        # log_print('=' * 100)

        # Layer 2 input -> output
        layerTwo = [1.0 for i in range(self.numOfLabels)]
        for i in range(self.numOfLabels):
            for prec in precOutput:
                layerTwo[i] *= prec[i]

        # log_print('Layer 2 output')
        # log_print('-' * len('Layer 1 output'))
        # log_print(layerTwo)
        # log_print('=' * 100)

        # Layer 3 input -> output
        l2_Sum = sum(layerTwo)
        layerThree = [float(x) / l2_Sum for x in layerTwo]
        # log_print('Layer 3 output')
        # log_print('-' * len('Layer 1 output'))
        # log_print(layerThree)
        # log_print('=' * 100)

        # Layer 4 input -> output
        layerFour = []
        for i in range(self.numOfLabels):
            fi = self.consequents[i].membership_degree(
                layerTwo[i],
                self.consParams[i]
            )
            layerFour.append(fi)
        # log_print('Layer 4 output')
        # log_print('-' * len('Layer 1 output'))
        # log_print(layerFour)
        # log_print('=' * 100)

        log_print('Done!')
        # log_print('Inferred phoneme is {}, target was {}'.format(
        #    step(result),
        #    expected)
        # )
        return layerFour, layerTwo, layerThree

    def update_consequents(self, l4Input, rsMatx, expected, zeta):
        """ Update the consequent parameters with a Least Square Estimation

        Parameters
        ----------
        l4Input : list of double
            The output from the second layer
        rsMatx : list of double
            The output
        expected : str
            The expected phoneme
        zeta : double
            A small double value to create a desired output. This will be used
            to update the outputs.

        Returns
        -------
        inference : double
            The crisp value of the output nodes, or the 'clip' operation.
        """
        layerSize = len(self.consParams)
        log_print('Updating consequent parameters...', end_='')
        # Coeficient matrix
        # Ones are at the second column because we are using a linear
        #
        coefMatrix = [
            [l4Input[i], 1] for i in range(layerSize)
        ]

        rsMatrix = [rsMatx for i in range(layerSize)]
        # log_print('Coefficient matrix is...\n{}\n'.format(array(coefMatrix)))

        self.consParams = lse(coefMatrix, rsMatrix, lamb=0.03)
        self.consParams = self.consParams.tolist()
        self.consParams = [[x, y] for (x, y) in zip(*self.consParams)]
        log_print('Done!')
        # log_print(
        #    'Done!\nNew parameters are...\n{}'.format(array(self.consParams))
        # )

    def backward_pass(self, err, inputs, layerFour, target):
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
        log_print('Initializing backward pass...')

        # Computing the derivative of each label
        derivatives = zeros(
            (self.numOfLabels, len(self.precedents[0].params[0]))
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
        # for i in range(fuzzSetDerivs.shape[0]):
        #     fuzzSetDerivs[i] = fuzzSetDerivs[i] * error

        log_print('Finished backward pass!')

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
        for (feature, expected) in trainingData:
            log_print('\n\nTraining for data \n{}'.format(feature))
            log_print('=' * 150)
            epoch = 0
            converged = False
            while epoch < nEpochs and not converged:
                l4, l2, l3 = self.forward_pass(feature, expected)
                self.update_consequents(l2, l4, expected, 1e-5)
                inference = self.inference.infer(l2, l4)
                pred_phn = self.alph.step(inference)
                error = (inference - self.alph.value(expected)) ** 2
                converged = pred_phn == expected or error <= errTolerance
                if not converged:
                    self.backward_pass(error, feature, l4, [])
                log_print(
                    'Error for {}-th epoch is {}'.format(epoch + 1, error)
                )
                epoch += 1
            log_print('Convergence occurred at epoch {}'.format(epoch))
