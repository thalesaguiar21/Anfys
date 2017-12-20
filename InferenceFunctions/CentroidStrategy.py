from InferenceFunctions.InferenceStrategy import InferenceStrategy


class CentroidStrategy(InferenceStrategy):
    """ This class is an Centroid based Inference Strategy. This will
    """

    def __init__(self):
        pass

    def infer(self, inputs, weights):
        """Find the centroid from the given set of values"""
        totalWeight = 0.0
        totalStrength = 0.0
        for w in weights:
            totalWeight += w
        for strength in inputs:
            totalStrength += strength
        return totalStrength / totalWeight
