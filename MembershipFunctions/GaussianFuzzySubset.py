from MembershipFunctions.FuzzySubset import FuzzySubset
from math import exp, sqrt, pi


class GaussianFuzzySubset(FuzzySubset):
    """This class implements a Gaussian Density Distribu Fuzzy Subset"""

    def __init__(self, begin=-0.5, end=0.5, mean=100, dev=12):
        """ Initializes a new Gaussian Fuzzy Subset with corresponding
        membership function as a Normal Distribution over the parameters
        given in initialization.

        Keyword arguments:
        begin   -- The X value where the function starts, default is 0
        end     -- The X value where the function ends, default is 1
        mean    -- Mean of the distribution, default is 100
        dev     -- Standard deviation of the distribution, default is 12
        """
        self.name = 'Gaussian Fuzzy Subset'
        self.begin = begin if begin > 0 else 0
        self.end = end if end > 0 else 1
        self.mean = mean
        self.dev = dev if dev > 0 else 0

    def membershipValue(self, value):
        """ Computes the Probability Density Function for a given value. This
        method also shifts the function to the specified values in the object
        construction.
        """
        value = value - (self.end + self.begin) / 2.0
        denom = sqrt(2.0 * pi * self.dev**2.0)
        num = exp((-(value - self.mean)**2) / (2.0 * self.dev**2))
        return num / denom
