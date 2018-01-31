from MembershipFunctions.LinguisticLabel import LinguisticLabel
from Errors import err
from math import exp


class GaussianTwo(LinguisticLabel):
    """This class implements a Gaussian Density Distribu Fuzzy Subset"""

    def __init__(self, name='Gaussian Two Label'):
        """ Initializes a new Gaussian Fuzzy Subset with corresponding
        membership function as a Normal Distribution over the parameters
        given in initialization.
        """
        LinguisticLabel.__init__(self)
        self.name = name

    def membership_degree(self, value, params):
        """ Compute the membership degree for the following bellshaped function:
        B(x) = exp(((x - c) / a)^2)
        """
        p = - ((value - params[2]) / params[0]) ** 2
        return exp(p)

    def derivative_at(self, value, var, params):
        result = 0
        denom = 1.0
        k = (value - params[2]) ** 2 / params[0] ** 2
        result = 2 * (value - params[2]) * exp(-k)
        if var == 'a':
            denom = params[0] ** 3
        elif var == 'b':
            result = 0
        elif var == 'c':
            denom = params[0] ** 2
        else:
            print err['INVALID_DERIV_ARG'].format(var)
        return result / denom
