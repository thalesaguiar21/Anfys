from MembershipFunctions.LinguisticLabel import LinguisticLabel


class GaussianThree(LinguisticLabel):
    """This class implements a Gaussian Density Distribu Fuzzy Subset"""

    def __init__(self):
        """ Initializes a new Gaussian Fuzzy Subset with corresponding
        membership function as a Normal Distribution over the parameters
        given in initialization.
        """
        self.name = 'Gaussian Fuzzy Subset'

    def membershipValue(self, value, premiseParams):
        """ Computes the Probability Density Function for a given value. This
        method also shifts the function to the specified values in the object
        construction.


        Keyword arguments:
        value -- Value for calculate the membership
        premiseParams -- Array of premise parameters
        """
        # (1 / ((x-ci)/ai)^2)^bi
        denom = 1 + ((
            (value - premiseParams[2]) /
            premiseParams[0]) ** 2) ** premiseParams[1]
        return 1 / denom

    def derivativeAt(self, value):
        pass
