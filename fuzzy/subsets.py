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
        variables = ['a', 'b', 'c'] if var is None else [var]
        derivs = []
        for line in params:
            for variable in variables:
                deriv = self.mem_func.derivative_at(value, variable, *line)
                derivs.append(deriv)
        return derivs
