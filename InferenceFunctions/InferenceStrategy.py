class InferenceStrategy:
    """ This class represent an abstract strategy for the inference layer of
    the Adaptive Neural Inference System.
    """
    def __init__(self):
        pass

    def infer(self, weights, values):
        """ This method is used to infer a a result from a given set of data.of
        In this case, for the ANFIS, this method will receive a list of inputs
        and select some of its data to infer something about the fire strength
        from the rules.

        Parameters
        ----------
        weights : list of double
            The weights associated with the inputs
        values : list of double
            Incoming inputs from the previous layer.


        Returns
        -------
        value : float
            A crisp value representing an 'clip', or centroid, of the
            incoming inputs

        Raises
        ------
        NotImplementedError
            If the subclass has not implemented this method.
        """
        raise NotImplementedError()
