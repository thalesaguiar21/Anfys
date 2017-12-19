import NotImplementedType


class FuzzySubset:
    """ This class represents a membership function for a given Fuzzy Subset.
    Fuzzy. Therefore, this can be seen as the Fuzzy Subset itself, since
    that is represented by how much an element belongs to a set. Usually
    this is implemented by using an bell shaped function (e.g. Gaussian
    function) for simplicity. However, any partially diferentiable function
    can be used as a membership function.

    You can inherit this class and overwrite the method membershipValue
    """

    def __init__(self):
        """Does nothing"""
        pass

    def membershipValue(self, value):
        """ This method is reponsible to calculate the membership of a given
        value. Here is where a partially diferentiable function must be used.

        Keyword arguments:
        value -- the element of a given set

        Return:
        value -- the membership value of the element

        Raise:
        NotImplementedType -- if the subclass has not implemented this method.
        """
        raise NotImplementedType()
