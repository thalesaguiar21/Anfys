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

    def __init__(self, pre, consequents, cons_params, alph):
        """ This method initialize a new instance of an ANFIS.

        Parameters
        ----------
        pre : list of FuzzySubset
            The first layer, or precedent membership functions
        consequents : list of LinguisticLabel
            The fourth layer, or the consequent membership functions
        cons_params : 2D list of double
            The parameters p and q of each consequent memebership function
        alph : MappedAlphabet
            The alphabet that will be used for phoneme recognition
        """
        self.__numOfLabels = len(pre[0].labels)
        self.cons_params = cons_params
        self.precedents = pre
        self.consequents = consequents
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
        layerTwo = [1.0 for i in range(self.__numOfLabels)]
        for i in range(self.__numOfLabels):
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

        self.update_consequents(layerTwo, layerThree)

        # Layer 4 input -> output
        layerFour = []
        for i in range(self.__numOfLabels):
            fi = self.consequents[i].membership_degree(
                layerTwo[i],
                self.cons_params[i]
            )
            layerFour.append(fi * layerThree[i])
        # log_print('Layer 4 output')
        # log_print('-' * len('Layer 1 output'))
        # log_print(layerFour)
        # log_print('=' * 100)

        log_print('Done!')
        # log_print('Inferred phoneme is {}, target was {}'.format(
        #    step(result),
        #    expected)
        # )
        return layerTwo, layerFour

    def update_consequents(self, layer2, layer3, lamb=0.03):
        """ Update the consequent parameters with a Least Square Estimation

        Parameters
        ----------
        layer2 : list of double
            The output from the second layer
        rsMatx : list of double
            The output

        Returns
        -------
        inference : double
            The crisp value of the output nodes, or the 'clip' operation.
        """
        layerSize = len(self.cons_params)
        log_print('Updating consequent parameters...', end_='')
        # Coeficient matrix
        # Ones are at the second column because we are using a linear
        #
        coefMatrix = [
            [layer2[i] * layer3[i], layer3[i]] for i in range(layerSize)
        ]
        # log_print('Coefficient matrix is...\n{}\n'.format(array(coefMatrix)))

        self.cons_params = lse(coefMatrix, self.alph._values, lamb)
        self.cons_params = self.cons_params.tolist()
        self.cons_params = [[x, y] for (x, y) in zip(*self.cons_params)]
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
        log_print('Initializing backward pass...')

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
        for i in range(fuzzSetDerivs.shape[0]):
            fuzzSetDerivs[i] = fuzzSetDerivs[i] * err

        log_print('Finished backward pass!')

    def __predict(self, crisp_value):
        smallest = abs(self.alph._values[0][0] - crisp_value)
        idx = 0
        valuesLen = len(self.alph._values)
        for i in range(1, valuesLen):
            tmp = abs(self.alph._values[i][i] - crisp_value)
            if tmp < smallest:
                smallest = tmp
                idx = i
        return self.alph._symbols[idx]

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
            log_print('\n\nTraining for data \n{}'.format((feature, expected)))
            log_print('=' * 150)
            epoch = 0
            converged = False
            while epoch < nEpochs and not converged:
                l2, l4 = self.forward_pass(feature, expected)
                pred_phn = self.__predict(sum(l4))
                error = 0
                for (target, output) in zip(self.alph[pred_phn], l4):
                    error += (target - output) ** 2
                converged = pred_phn == expected or error <= errTolerance
                log_print(
                    'Predicted {} and expected {}'.format(pred_phn, expected)
                )
                if not converged:
                    self.backward_pass(error, feature, l4, [])
                log_print(
                    'Error for {}-th epoch is {}\n'.format(epoch + 1, error)
                )
                epoch += 1
            log_print('Convergence occurred at epoch {}'.format(epoch))
