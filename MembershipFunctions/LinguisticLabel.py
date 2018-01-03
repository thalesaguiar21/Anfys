class LinguisticLabel:
    """This class represents a Linguistic Label for a Fuzzy Subset."""

    def __init__(self):
        """Does nothing"""
        self.name = 'Linguistic Label Subset'

    def membershipValue(self, value, premiseParams):
        """ This method is reponsible to calculate the membership of a given
        value. Here is where a piecewise diferentiable function must be used.

        Keyword arguments:
        value -- the element of a given set
        premiseParams -- Set of premise parameters

        Return:
        value -- the membership value of the element

        Raise:
        NotImplementedError -- if the subclass has not implemented this method.
        """
        raise NotImplementedError()

    def derivativeAt(self, value):
        """ Calculate the derivative of this function at the given point.

        Keyword arguments:
        value -- Point to compute the derivative.

        Raise:
        NotImplementedError -- If the subclass has not implemented this method.
        """
        raise NotImplementedError()
