from MembershipFunction import MembershipFunction
from math import log
from Errors import err


class BellThree(MembershipFunction):
    """ This class represents a Bell Shaped function with three parameters """

    min_denom = 1e-15

    def membership_value(value, a, b, c=None):
        """
        """
        if((a is None) or (b is None) or (c is None)):
            raise ValueError("Gaussian three function needs exact three arg \
                uments, less where given!")
        if(a > -1 and a < 1):
            if(a > -BellThree.min_denom and a < BellThree.min_denom):
                raise ValueError("Value of parameter \"a\" is too small!")

        denom = 1.0 + (((value - c) / a) ** 2.0) ** b
        return 1.0 / denom

    def derivative_at(value, var, a, b, c=None):
        result = 0

        if((a is None) or (b is None) or (c is None)):
            raise ValueError("Gaussian three function needs exact three arg \
                uments, less where given!")

        k = (value - c) ** 2 / a ** 2
        if var == 'a':
            result = 2 * b * k ** 2
            result /= a * ((k ** b) + 1) ** 2
        elif var == 'b':
            result = (k ** b) * log(k)
            result /= ((k ** b) + 1) ** 2
        elif var == 'c':
            result = 2 * b * (value - c)
            result *= k ** (b - 1)
            result /= a ** 2 * ((k ** b) + 1) ** 2
        else:
            print err['INVALID_DERIV_ARG'].format(var)
        return result
