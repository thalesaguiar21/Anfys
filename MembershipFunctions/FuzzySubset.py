class FuzzySubset():
    """This class represents a Fuzzy Subset with n linguistic labels."""

    def __init__(self, labels, params):
        """ Create a Fuzzy subset with the given labels and parameters.

        Parameters
        ----------
        labels : [LinguistcLabel]
            A list of Linguistic Labels
        Params : 2D array
            A 2D array of dimension Nx3, where N is the number of Labels
        """
        self.labels = labels
        self.params = params

    def evaluate(self, value):
        """ Compute the membership of the given value. That is, how much
        the given value belongs to each of the linguistic labels

        Parameters
        ----------
        value : float
            The value to compute the membership degree.

        Returns
        -------
        mem_degree : [float]
            A list with the membership of value for each Linguistic Label of
            this Fuzzy Subset.
        """
        mem_degree = []
        for i in range(len(self.labels)):
            mem_degree.append(
                self.labels[i].membershipDegree(value, self.params[i])
            )
        return mem_degree
