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


class FuzzySet:
    """ This class represents a set of fuzzy membership functions """

    def __init__(self, mem_func):
        self.mem_func = mem_func

    def evaluate(self, value, params):
        """ Evaluate the given value in the current subset of membership
        functions

        Parameters
        ----------
        value : double
            The value to be evaluated
        params : numpy.arr of double
            A 2D array with the parameters of each membership function.

        Returns
        -------
        mem_degree : numpy.arr of double
            An array with the membership degree of the given value with respect
            to each membership function in this set.
        """
        mem_degree = []
        for line in params:
            mem_degree.append(self.mem_func.membership_degree(value, *line))
        return mem_degree

    def derivs_at(self, value, var, params):
        """ Compute the derivative of each membership function in this set at
        the given value with respect to a given variable.

        Parameters
        ----------
        value : double
            The derivative argument
        var : string
            A string with the variable to compute the partial derivative
        params : numpy.arr of double
            A 2D array with the parameters of each membership function.

        Returns
        -------
        derivs : numpy.arr of double
            An array with the value of the derivative at the given value for
            each membership function in this set.
        """
        variables = ['a', 'b', 'c'] if var is None else var
        derivs = []
        for line in params:
            derivs.append(
                self.mem_func.derivative_at(
                    value, variable, *line
                ) for variable in variables
            )
        return derivs
