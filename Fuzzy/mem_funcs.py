from math import log, exp


class MembershipFunction:
    """ This class represents an interface for Membership Functions. Any
    function class must have both methods to calculate the membership
    degree of a given value, and to calculate the derivative at a point
    with respect to a variable.
    """

    def __init__(self):
        pass

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

    def derivative_at(self, value, var, a, b, c=None):
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


class BellThree(MembershipFunction):
    """ This class represents a Bell Shaped function with three parameters """

    min_denom = 1e-15

    def __init__(self):
        pass

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
        """
        if((a is None) or (b is None) or (c is None)):
            raise ValueError("Gaussian three function needs exact three arg \
                uments, less where given!")
        if(a > -1 and a < 1):
            if(a > -BellThree.min_denom and a < BellThree.min_denom):
                raise ValueError("Value of parameter \"a\" is too small!")

        denom = 1.0 + (((value - c) / float(a)) ** 2.0) ** b
        return 1.0 / denom

    def derivative_at(self, value, var, a, b, c=None):
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
        """
        result = 0

        if((a is None) or (b is None) or (c is None)):
            raise ValueError("Gaussian three function needs exact three arg \
                uments, less where given!")

        k = (value - c) ** 2 / float(a) ** 2
        if var == 'a':
            result = 2.0 * b * k ** 2
            result /= a * ((k ** b) + 1) ** 2
        elif var == 'b':
            result = (k ** b) * log(k)
            result /= ((k ** b) + 1) ** 2
            result = result * (-1)
        elif var == 'c':
            result = -2.0 * b * k ** b
            result /= (c - value) * ((k ** b) + 1) ** 2
        else:
            raise Warning('Function has no variable \'{}\''.format(var))
        return result


class BellTwo(MembershipFunction):
    """ This class represents a Bell Shaped function with two parameters """

    def __init__(self):
        pass

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
        """
        arg = - ((value - b) / float(a)) ** 2
        return exp(arg)

    def derivative_at(self, value, var, a, b, c=None):
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
        k = (value - b) ** 2 / float(a) ** 2
        if var == 'a':
            result = 2 * ((value - b) ** 2) * exp(-k)
            denom = a ** 3
        elif var == 'b':
            result = 2 * (value - b) * exp(-k)
            denom = a ** 2
        else:
            raise Warning('Function has no variable \'{}\''.format(var))
        return result / denom


class DiscreteSigmoid(MembershipFunction):
    """ This class implements a discrete Sigmoid function, where two
    parameters (p, q) are used as a linear function for approximate the
    sigmoid curve.
    """

    def __init__(self):
        pass

    def membership_degree(self, value, a, b, c=None):
        """ Computes the memebership degree of the given value for an
        discrete sigmoid function, described by two parameters.
        """
        mem_degree = 0
        if value <= a:
            mem_degree = 0.001
        elif a < value and b > value:
            a = 0.999 / b - a
            b = 0.001 - a * value
            mem_degree = a * value + b
        else:
            mem_degree = 1.0

        return mem_degree

    def derivative_at(self, value, var, a, b, c=None):
        """ Computes the derivative with respect to one of the params, for this
        function at the given value.

        Parameters
        ----------
        value : float
            The value to be computed.
        var : chr
            A character of the variable to derivate.

        Returns
        -------
        value : float
            The value of the derivative with respect to var at the given value.
        """
        dF_dAlpha = 0
        if var == 'a':
            dF_dAlpha = 1 / a
        elif var == 'b':
            dF_dAlpha = 1
        elif var == 'c':
            dF_dAlpha = 0
        else:
            raise Warning('Function has no variable \'{}\''.format(var))
        return dF_dAlpha


class PiecewiseLogit(MembershipFunction):
    """ A piecewise linear approximation of logit (inverse of sigmoid) function
    with two parameters (p, q).
    """
    def __init__(self):
        self.__low = 1e-15
        self.__high = 0.99999
        self.__hl = self.__high - self.__low

    def membership_degree(self, value, a, b, c=None):
        # a = Pmin, b = Pmax
        mem_degree = 0
        lin_coef = (b - a) / self.__hl
        indep = a - lin_coef
        if value <= self.__low:
            mem_degree = a
        elif self.__low < value and self.__high > value:
            mem_degree = lin_coef * value + indep
        elif value >= self.__high:
            mem_degree = b
        return mem_degree

    def derivative_at(self, value, var, a, b, c=None):
        result = 0.0
        if value < self.__low or value > self.__high:
            return 0.0
        elif var == 'a':
            numerator = - value + 1 - self.__hl
            return float(numerator) / self.__hl
        elif var == 'b':
            return (float(value) + 1.0) / self.__hl
        else:
            raise Warning('Function has no variable \'{}\''.format(var))
        return result
