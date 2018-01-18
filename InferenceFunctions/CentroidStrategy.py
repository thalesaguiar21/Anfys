from InferenceFunctions.InferenceStrategy import InferenceStrategy


class CentroidStrategy(InferenceStrategy):
    """ This class is an Centroid based Inference Strategy. This will
    """

    def __init__(self):
        InferenceStrategy.__init__(self)

    def infer(self, weights, values):
        """Find the centroid from the given set of values"""
        return sum(values)
