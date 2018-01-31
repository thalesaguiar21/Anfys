class LinguisticLabel:
    """This class represents a Linguistic Label for a Fuzzy Subset."""

    def __init__(self):
        """ Create an linguistic label, or a membership function """
        self.name = 'Linguistic Label Subset'

    def membership_degree(self, value, params):
        """ This method is reponsible to calculate the membership of a given
        value. Here is where a piecewise diferentiable function must be used.

        Parameters
        ----------
        value : float
            The element of a given set
        params : list of double
            An list of float parameters

        Return
        ------
        value : float
            The membership value of the element

        Raises
        ------
        NotImplementedError
            if the subclass has not implemented this method.
        """
        raise NotImplementedError()

    def derivative_at(self, value, var, params):
        """ Calculate the derivative of this function at the given point.

        Parameters
        ----------
        value : float
            Point to compute the derivative.
        var : chr
            The variable to partial derivate

        Returns
        -------
        value : float
            The value of the partial derivative with respect to the given
            variable at the point.

        Raises
        ------
        NotImplementedError
            If the subclass has not implemented this method.
        """
        raise NotImplementedError()
