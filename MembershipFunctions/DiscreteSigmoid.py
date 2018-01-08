from MembershipFunctions.LinguisticLabel import LinguisticLabel


class DiscreteSigmoid(LinguisticLabel):
    """ This class implements a discrete Sigmoid function, where two
    parameters (p, q) are used as a linear function for approximate the
    sigmoid curve.
    """

    def __init__(self):
        self.a = 1
        self.b = 0

    def membershipDegree(self, value, params):
        """ Computes the memebership degree of the given value for an
        discrete sigmoid function, described by two parameters.
        """
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

    def derivativeAt(self, value, var):
        """ Computes the derivative with respect to one of the params, for this
        function at the given value.

        Parameters
        ----------
        value : float
            The value to be computed.
        var : chr
            A character of the variable to derivate.

        Returns
        -------
        value : float
            The value of the derivative with respect to var at the given value.
        """
        return 1 / self.a
