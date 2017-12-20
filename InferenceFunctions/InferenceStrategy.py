class InferenceStrategy:
    """ This class represent an abstract strategy for the inference layer of
    the Adaptive Neural Inference System.
    """
    def __init__(self):
        pass

    def infer(self, inputs, weights):
        """ This method is used to infer a a result from a given set of data.of
        In this case, for the ANFIS, this method will receive a list of inputs
        and select some of its data to infer something about the fire strength
        from the rules.

        Keyword arguments:
        inputs -- A list of incoming inputs from the previous layer.
        weights -- A list of weights from the previous layer to inferece layer.

        Raise:
        NotImplementedError -- If the subclass has not implemented this method.

        Return:
        A value representing a valid conclusion for the problem.
        """
        raise NotImplementedError()
