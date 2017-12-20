from MembershipFunctions.FuzzySubset import FuzzySubset
from math import floor


class AbsoluteFuzzySubset(FuzzySubset):
    """ This class represents a Absolute function, that is, an Absolute Fuzzy
    subset.
    """

    def __init__(self, begin, end):
        """ Create an instance of an Absolute Fuzzy Subset. By specifying the
        begin and end of the function, that is, the limits.

        Keyword arguments:
        limit -- The range of the function.
        """
        self.name = 'Absolute Fuzzy Subset'
        self.begin = begin
        self.end = end

    def membershipValue(self, value):
        """ Computes the membership value for a given element. This will be
        using an absolute function in an specified range.

        Keyword arguments:
        value -- The element to compute the membership value.
        """
        limit = (self.end + self.begin) / 2.0
        value = floor(value - limit)
        if value >= self.begin and value <= self.end:
            if value < 0:
                return -value
            else:
                return value
        else:
            return 0
