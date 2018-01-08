from MembershipFunctions.LinguisticLabel import LinguisticLabel
from math import log
from Errors import err


class GaussianThree(LinguisticLabel):
    """This class implements a Gaussian Density Distribu Fuzzy Subset"""

    def __init__(self):
        """ Initializes a new Gaussian Fuzzy Subset with corresponding
        membership function as a Normal Distribution over the parameters
        given in initialization.
        """
        self.name = 'Gaussian Fuzzy Subset'
        self.params = []

    def membership_degree(self, value, premiseParams):
        """ Computes the Probability Density Function for a given value. This
        method also shifts the function to the specified values in the object
        construction.
        (1 / ((x-ci)/ai)^2)^bi
        """
        self.params = premiseParams
        denom = 1 + ((
            (value - premiseParams[2]) /
            premiseParams[0]) ** 2) ** premiseParams[1]
        return 1 / denom

    def derivative_at(self, value, var):
        result = 0
        k = (value - self.params[2]) ** 2 / self.params[0] ** 2
        if var == 'a':
            result = 2 * self.params[1] * (value - self.params[2]) ** 2
            result *= k ** (self.params[1] - 1)
            result /= self.params[0] ** 3 * (k ** self.params[1] + 1) ** 2
        elif var == 'b':
            result = (k ** self.params[1]) * log(k)
            result *= 1 / (k ** self.params[1] + 1) ** 2
        elif var == 'c':
            result = 2 * self.params[1] * (value - self.params[2])
            result *= k ** (self.params[1] - 1)
            result /= self.params[0] ** 2 * (k ** self.params[1] + 1) ** 2
        else:
            print err['INVALID_DERIV_ARG'].format(var)
        return result
