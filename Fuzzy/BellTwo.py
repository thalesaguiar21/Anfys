from MembershipFunction import MembershipFunction
from math import exp


class BellTwo(MembershipFunction):
    """
    """

    def __init__(self):
        pass

    def membership_degree(self, value, a, b, c=None):
        arg = - ((value - b) / float(a)) ** 2
        return exp(arg)

    def derivative_at(self, value, var, a, b, c=None):
        result = 0
        denom = 1.0
        k = (value - b) ** 2 / float(a) ** 2
        if var == 'a':
            result = 2 * ((value - b) ** 2) * exp(-k)
            denom = a ** 3
        elif var == 'b':
            result = 2 * (value - b) * exp(-k)
            denom = a ** 2
        elif var == 'c':
            result = 0
        else:
            print 'ERROR'
        return result / denom
