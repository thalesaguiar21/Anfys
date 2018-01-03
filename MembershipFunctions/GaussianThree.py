from MembershipFunctions.FuzzySubset import FuzzySubset


class GaussianThree(FuzzySubset):
    """This class implements a Gaussian Density Distribu Fuzzy Subset"""

    def __init__(self, linguisticLabels):
        """ Initializes a new Gaussian Fuzzy Subset with corresponding
        membership function as a Normal Distribution over the parameters
        given in initialization.


        Keyword arguments:
        begin -- The X value where the function starts, default is 0
        end -- The X value where the function ends, default is 1
        mean -- Mean of the distribution, default is 100
        dev -- Standard deviation of the distribution, default is 12
        """
        self.name = 'Gaussian Fuzzy Subset'
        self.linguisticLabels = linguisticLabels

    def membershipValue(self, x, premiseParams):
        """ Computes the Probability Density Function for a given value. This
        method also shifts the function to the specified values in the object
        construction.


        Keyword arguments:
        x -- Value for calculate the membership
        premiseParams -- Array of premise parameters
        """
        # (1 / ((x-ci)/ai)^2)^bi
        denom = 1 + ((
            (x - premiseParams[2]) /
            premiseParams[0]) ** 2) ** premiseParams[1]
        return 1 / denom

    def derivativeAt(self, value):
        pass
