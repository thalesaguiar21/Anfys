from MembershipFunction import MembershipFunction
from math import log


class BellThree(MembershipFunction):
    """ This class represents a Bell Shaped function with three parameters """

    min_denom = 1e-15

    def __init__(self):
        pass

    def membership_degree(self, value, a, b, c=None):
        """
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
            print 'ERROR'
        return result
