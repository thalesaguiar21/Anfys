from MembershipFunctions.LinguisticLabel import LinguisticLabel
from math import log
from Errors import err


class GaussianThree(LinguisticLabel):
    """This class implements a Gaussian Density Distribu Fuzzy Subset"""

    def __init__(self, name='Gaussian Fuzzy Subset'):
        """ Initializes a new Gaussian Fuzzy Subset with corresponding
        membership function as a Normal Distribution over the parameters
        given in initialization.
        """
        LinguisticLabel.__init__(self)
        self.name = name

    def membership_degree(self, value, premise_params):
        """ Computes the Probability Density Function for a given value. This
        method also shifts the function to the specified values in the object
        construction.
        """
        denom = 1 + ((
            (value - premise_params[2]) /
            premise_params[0]) ** 2) ** premise_params[1]
        return 1 / denom

    def derivative_at(self, value, var, params):
        result = 0
        if value - params[2] == 0:
            print 'Value is euqal to a param!'
            value += 1e-10
            raise Exception

        k = (value - params[2]) ** 2 / params[0] ** 2
        if var == 'a':
            result = 2 * params[1] * (value - params[2]) ** 2
            result *= k ** (params[1] - 1)
            result /= params[0] ** 3 * ((k ** params[1]) + 1) ** 2
        elif var == 'b':
            result = (k ** params[1]) * log(k)
            result *= 1 / (k ** params[1] + 1) ** 2
        elif var == 'c':
            result = 2 * params[1] * (value - params[2])
            result *= k ** (params[1] - 1)
            result /= params[0] ** 2 * ((k ** params[1]) + 1) ** 2
        else:
            print err['INVALID_DERIV_ARG'].format(var)
        return result
