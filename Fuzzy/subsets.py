class FuzzySubset():
    """This class represents a Fuzzy Subset with n linguistic labels."""

    def __init__(self, num_of_labels, params, mem_function):
        """ Create a Fuzzy subset with the given labels and parameters.

        Parameters
        ----------
        num_of_labels : int
            Number of linguistic labels of this fuzzy subset
        params : 2D list of double
            A 2D array of dimension Nx3, where N is the number of Labels
        mem_function : MembershipFunction
            The type of membership function to be used in this Fuzzy subset
        """
        self._num_of_labels = 1 if num_of_labels < 0 else num_of_labels
        self.params = params
        self.mem_function = mem_function

    def evaluate(self, value):
        """ Compute the membership of the given value. That is, how much
        the given value belongs to each of the linguistic labels

        Parameters
        ----------
        value : float
            The value to compute the membership degree.

        Returns
        -------
        mem_degree : list of double
            A list with the membership of value for each Linguistic Label of
            this Fuzzy Subset.
        """
        mem_degree = []
        for i in range(self._num_of_labels):
            mem_degree.append(
                self.mem_function.membership_degree(value, *self.params[i])
            )
        return mem_degree

    def derivs_at(self, value):
        derivs = []
        for i in range(self._num_of_labels):
            tmp = [
                self.mem_function.derivative_at(value, var, *self.params[i])
                for var in ['a', 'b', 'c']
            ]
            derivs.append(tmp)
        return derivs
