class FuzzySubset():
    """This class represents a Fuzzy Subset with n linguistic labels."""

    def __init__(self, labels, params):
        """

        Keyword arguments:
        labels -- A list of Linguistic Labels
        params -- A 2D array with the parameters
        """
        self.labels = labels
        self.params = params

    def evaluate(self, value):
        """ Compute the memebership of the given value.

        Keyword arguments:
        value -- The value to compute the membership.

        Return:
        A list with the membership of value for each Linguistic Label of this
        Fuzzy Subset.
        """
        result = []
        for i in range(len(self.labels)):
            result.append(
                self.labels[i].membershipValue(value, self.params[i])
            )
        return result
