import pdb
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
        self.params = []

    def membership_degree(self, value, premiseParams):
        """ Computes the Probability Density Function for a given value. This
        method also shifts the function to the specified values in the object
        construction.
        (1 / (1 + ((x-ci)/ai)^2)^bi)
        """
        denom = 1 + ((
            (value - premiseParams[2]) /
            premiseParams[0]) ** 2) ** premiseParams[1]
        return 1 / denom

    def derivative_at(self, value, var, params):
        # pdb.set_trace()
        result = 0
        if abs(params[0]) <= 1e-25:
            print 'ERROR! Param too small! A = ' + str(params[0])
        if value - params[2] == 0:
            value += 1e-10
        k = (value - params[2]) ** 2 / params[0] ** 2
        if var == 'a':
            result = 2 * params[1] * (value - params[2]) ** 2
            result *= k ** (params[1] - 1)
            # pdb.set_trace()
            # print('Result:\t' + str(result))
            # print('Params:\t' + str(params))
            # print('K:\t' + str(k))
            result /= params[0] ** 3 * (k ** params[1] + 1) ** 2
        elif var == 'b':
            result = (k ** params[1]) * log(k)
            result *= 1 / (k ** params[1] + 1) ** 2
        elif var == 'c':
            result = 2 * params[1] * (value - params[2])
            result *= k ** (params[1] - 1)
            result /= params[0] ** 2 * (k ** params[1] + 1) ** 2
        else:
            print err['INVALID_DERIV_ARG'].format(var)
        return result
