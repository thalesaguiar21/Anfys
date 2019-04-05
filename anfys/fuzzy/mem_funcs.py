import math
import pdb

MIN_MEMBERSHIP = 1e-10


class MembershipFunction:
    """ This class represents an interface for Membership Functions. Any
    function class must have both methods to calculate the membership
    degree of a given value, and to calculate the derivative at a point
    with respect to a variable.
    """

    def __init__(self):
        self.parameters = None
        self.qtd_params = 0

    def membership_degree(self, value, a, b, c=None):
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

    def partial(self, value, var, a, b, c=None):
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

        Returns
        -------
        deriv : double
            The result of computing the derivative of the Bell Two function
            at the given value.
        """
        raise NotImplementedError()


class BellThree(MembershipFunction):
    """ This class represents a Bell Shaped function with three parameters """

    min_denom = 1e-15

    def __init__(self):
        self.parameters = ['a', 'b', 'c']
        self.qtd_params = 3

    def membership_degree(self, value, a, b, c=None):
        """ This method computes the membership degree of the given value
        accordingly to the function defined by the parameters.

        Paramaters
        ----------
        value : double
            The value to compute the memebership degree
        a : double
            The fisrt parameter
        b : double
            The second parameter
        c : double
            The third parameter. Not used on BellTwo. Defaults to None

        Returns
        -------
        degree : double
            A number between 0 and 1 representing the membership degree of the
            value in this Fuzzy Subset.

        Raises
        ------
        ValueError
            If any given argument is None
        ZeroDivisionError
            If a is 0
        """
        if value is None or a is None or b is None or c is None:
            raise ValueError("Gaussian three function needs exact three arg \
                uments, less where given!")
        if a == 0:
            raise ValueError('Parameter a was 0 at MF mem degree')

        tmp1 = (value - c) / a
        denom = 1.0 + (tmp1 ** 2.0) ** b
        return 1.0 / denom

    def partial(self, value, var, a, b, c=None):
        """ Compute the derivative at the given point (value) with respect to
        a variable.

        Parameters
        ----------
        value : double
            THe value to compute the membership degree
        a : double
            The first parameter
        b : double
            The second parameter
        c : double
            The third parameter. Defaults to None.

        Returns
        -------
        deriv : double
            The result of computing the derivative of the Bell Two function
            at the given value.

        Raises
        ------
        ValueError
            If any given argument is None
        """
        result = 0

        if value is None or a is None or b is None or c is None:
            raise ValueError("Gaussian three function needs exact three arg \
                uments, less where given!")
        if a == 0:
            raise ValueError('Parameter a was 0 in MF deriv')

        tmp1 = (value - c) / a
        tmp2 = (tmp1 ** 2) ** b
        denom = (1 + tmp2) ** 2
        if var == 'a':
            result = 2.0 * b * tmp2 / (a * denom)
        elif var == 'b':
            result = (- tmp2 * math.log(tmp1 ** 2)) / denom
        elif var == 'c':
            result = 2.0 * b * (value - c) * tmp1 ** (2.0 * b - 2.0)
            result /= denom * a ** 2.0
        return result


class BellTwo(MembershipFunction):
    """ This class represents a Bell Shaped function with two parameters """

    def __init__(self):
        self.parameters = ['a', 'b']
        self.qtd_params = 2

    def membership_degree(self, value, a, b, c=None):
        """ Computes the membership degree of the given value, with respect to
        this membership function and the given parameters.

        Parameters
        ----------
        value : double
            THe value to compute the membership degree
        a : double
            The first parameter
        b : double
            The second parameter
        c : double
            The third parameter. Defaults to None, notice that this is not
            used in this class, even though you pass any value to it.

        Returns
        -------
        degree : double
            A number between 0 and 1 representing the membership degree of the
            given value for this fuzzy subset.

        Raises
        ------
        TypeError
            Case an argument, except c, is None
        ZeroDivisionError
            Case 'a' is zero
        """
        arg = - ((value - b) / a) ** 2
        return max(math.exp(arg), MembershipFunction.MIN_MEMBERSHIP)

    def partial(self, value, var, a, b, c=None):
        """ Compute the derivative at the given point (value) with respect to
        a variable.

        Parameters
        ----------
        value : double
            THe value to compute the membership degree
        a : double
            The first parameter
        b : double
            The second parameter
        c : double
            The third parameter. Defaults to None, notice that this is not
            used in this class, even though you pass any value to it.

        Returns
        -------
        deriv : double
            The result of computing the derivative of the Bell Two function
            at the given value.
        """
        result = 0
        denom = 1.0
        k = (value - b) ** 2 / a ** 2
        if var == 'a':
            result = 2 * ((value - b) ** 2) * math.exp(-k)
            denom = a ** 3
        elif var == 'b':
            result = 2 * (value - b) * math.exp(-k)
            denom = a ** 2
        else:
            raise ValueError('BellTwo has no parameter \'{}\''.format(var))
        return result / denom


class PiecewiseLogit(MembershipFunction):
    """ A piecewise linear approximation of logit (inverse of sigmoid) function
    with two parameters (p, q).
    """

    def __init__(self):
        self.__low = 1e-8
        self.__high = 1.0 - self.__low
        self.__hl = self.__high - self.__low
        self.parameters = ['a', 'b']
        self.qtd_params = 2

    def membership_degree(self, value, a, b, c=None):
        """ Computes the membership degree of the given value, with respect to
        this membership function and the given parameters.

        Parameters
        ----------
        value : double
            The value to compute the membership degree
        a : double
            The first parameter
        b : double
            The second parameter
        c : double
            The third parameter. Defaults to None, notice that this is not
            used in this class, even though you pass any value to it.

        Returns
        -------
        degree : double
            A number between 0 and 1 representing the membership degree of the
            given value for this fuzzy subset.
        """
        # a = Pmin, b = Pmax
        mem_degree = 0
        if value <= self.__low:
            mem_degree = a
        elif self.__low < value and self.__high > value:
            mem_degree = (value - self.__low) * b + (self.__high - value) * a
            mem_degree = mem_degree / self.__hl
        elif value >= self.__high:
            mem_degree = b
        return mem_degree

    def partial(self, value, var, a, b, c=None):
        """ Compute the derivative at the given point (value) with respect to
        a variable.

        Parameters
        ----------
        value : double
            THe value to compute the membership degree
        a : double
            The first parameter
        b : double
            The second parameter
        c : double
            The third parameter. Defaults to None, notice that this is not
            used in this class, even though you pass any value to it.

        Returns
        -------
        deriv : double
            The result of computing the derivative of the Logit function
            at the given value.
        """
        result = 0.0
        if value < self.__low or value > self.__high:
            return 0.0
        elif var == 'a':
            numerator = - value + 1 - self.__hl
            return numerator / self.__hl
        elif var == 'b':
            return (value + 1.0) / self.__hl
        return result

    def build_sys_term(self, value, weight):
        return [weight * (value - self.__low) / self.__hl,
                weight * (self.__high - value) / self.__hl]
