from MembershipFunctions.LinguisticLabel import LinguisticLabel
from math import exp


class GaussianTwo(LinguisticLabel):
    """This class implements a Gaussian Density Distribu Fuzzy Subset"""

    def __init__(self):
        """ Initializes a new Gaussian Fuzzy Subset with corresponding
        membership function as a Normal Distribution over the parameters
        given in initialization.
        """
        self.name = 'Gaussian Linguistic Label'

    def membershipValue(self, value, premiseParams):
        """
        """
        p = - ((value - premiseParams[2]) / premiseParams[0]) ** 2
        return exp(p)

    def derivativeAt(self, value):
        pass
