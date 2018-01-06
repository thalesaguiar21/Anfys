from MembershipFunctions.LinguisticLabel import LinguisticLabel


class DiscreteSigmoid(LinguisticLabel):
    """ This class implements a discrete Sigmoid function, where two
    parameters (p, q) are used as a linear function for approximate the
    sigmoid curve.
    """

    def __init__(self):
        self.a = 1
        self.b = 0

    def membershipValue(self, value, params):
        self.a = 1 / (params[1] - params[0])
        self.b = 1 - (params[1] * self.a)
        result = 0
        if value >= 1:
            result = params[1]
        elif value < 1 and value > 0:
            result = (value - self.b) / self.a
        else:
            result = params[0]

        return result

    def derivativeAt(self, value):
        return 1 / self.a
