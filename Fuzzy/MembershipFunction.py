class MembershipFunction:
    """ This class represents an interface for Membership Functions. Any
    function class must have both methods to calculate the membership
    degree of a given value, and to calculate the derivative at a point
    with respect to a variable.
    """
    def membership_value(value, a, b, c=None):
        """ Computes the membership degree of the given value

        Parameters
        ----------
        value : double
            The value to be checked
        a : double
            The first parameter
        b : double
            The second parameter
        c : double
            The third parameter. Defaults to None

        Returns
        -------
        degree : double
            A value between 0 and 1 representing the memebership degree of the
            given value.
        """
        raise NotImplementedError()

    def derivative_at(value, var, a, b, c=None):
        """ Computes the derivative of the membership function with respect to
        a variable, at the given point.

        Parameters
        ----------
        value : double
            The point where the derivative should be computed
        a : double
            The first parameter
        b : double
            The second parameter
        c : double
            The third parameter. Defaults to None.
        """
        raise NotImplementedError()
