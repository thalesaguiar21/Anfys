from MembershipFunctions.LinguisticLabel import LinguisticLabel
from Errors import err
from math import exp


class GaussianTwo(LinguisticLabel):
    """This class implements a Gaussian Density Distribu Fuzzy Subset"""

    def __init__(self):
        """ Initializes a new Gaussian Fuzzy Subset with corresponding
        membership function as a Normal Distribution over the parameters
        given in initialization.
        """
        self.name = 'Gaussian Linguistic Label'
        self.params = []

    def membershipDegree(self, value, premiseParams):
        """ Compute the membership degree for the following bellshaped function:
        B(x) = exp(((x - c) / a)^2)
        """
        self.params = premiseParams
        p = - ((value - premiseParams[2]) / premiseParams[0]) ** 2
        return exp(p)

    def derivativeAt(self, value, var):
        result = 0
        denom = 1.0
        k = (value - self.params[2]) ** 2 / self.params[0] ** 2
        result = 2 * (value - self.params[2]) * exp(-k)
        if var == 'a':
            denom = self.params[0] ** 3
        elif var == 'c':
            denom = self.params[0] ** 2
        else:
            print err['INVALID_DERIV_ARG'].format(var)
        return result / denom
