from InferenceStrategy import InferenceStrategy


class WeightedAverage(InferenceStrategy):
    """ Computes the weighted average value given the weights and values """
    def __init__(self):
        InferenceStrategy.__init__(self)

    def infer(self, weights, values):
        result = -1
        if weights is not None:
            numerator = 0
            for (w, v) in zip(weights, values):
                numerator += w * v
            result = numerator / sum(weights)
        return result
