from MembershipFunctions.FuzzySubset import LinguisticLabel
from math import exp


class GaussianTwo(LinguisticLabel):
    """This class implements a Gaussian Density Distribu Fuzzy Subset"""

    def __init__(self):
        """ Initializes a new Gaussian Fuzzy Subset with corresponding
        membership function as a Normal Distribution over the parameters
        given in initialization.


        Keyword arguments:
        begin -- The X value where the function starts, default is 0
        end -- The X value where the function ends, default is 1
        mean -- Mean of the distribution, default is 100
        dev -- Standard deviation of the distribution, default is 12
        """
        self.name = 'Gaussian Linguistic Label'

    def membershipValue(self, value, premiseParams):
        """ Computes the Probability Density Function for a given value. This
        method also shifts the function to the specified values in the object
        construction.
        """
        p = - ((value - premiseParams[2]) / premiseParams[0]) ** 2
        return exp(p)

    def derivativeAt(self, value):
        pass
