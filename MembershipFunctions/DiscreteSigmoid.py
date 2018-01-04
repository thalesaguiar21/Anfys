from MembershipFunctions.LinguisticLabel import LinguisticLabel


class DiscreteSigmoid(LinguisticLabel):
    """
    """

    def __init__(self):
        self.a = 1
        self.b = 0
        pass

    def membershipValue(self, value, params):
        a = 1 / (params[1] - params[0])
        b = 1 - (params[1] * a)
        result = 0
        if value >= 1:
            result = params[1]
        elif value < 1 and value > 0:
            result = (value - b) / a
        else:
            result = params[0]

        return result

    def derivativeAt(self, value):
        return 1 / self.a
