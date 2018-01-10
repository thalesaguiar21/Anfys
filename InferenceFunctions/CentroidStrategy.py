from InferenceFunctions.InferenceStrategy import InferenceStrategy


class CentroidStrategy(InferenceStrategy):
    """ This class is an Centroid based Inference Strategy. This will
    """

    def __init__(self):
        InferenceStrategy.__init__(self)
        pass

    def infer(self, inputs):
        """Find the centroid from the given set of values"""
        return sum(inputs)
