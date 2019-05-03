import math
from abc import ABC, abstractmethod
import numpy as np
from anfys.lse import clip

MIN_MEMBERSHIP = 1e-10


class MembershipFunction(ABC):
    """ This class represents an interface for Membership Functions. Any
    function class must have both methods to calculate the membership
    degree of a given value, and to calculate the derivative at a point
    with respect to a variable.
    """

    def __init__(self):
        self.parameters = None
        self.qtd_params = 0

    @abstractmethod
    def membership_degree(self, value, a, b, c=None):
        pass

    @abstractmethod
    def partial(self, value, var, a, b, c=None):
        pass


class BellThree(MembershipFunction):
    """ This class represents a Bell Shaped function with three parameters """

    min_denom = 1e-15

    def __init__(self):
        self.parameters = ['a', 'b', 'c']
        self.qtd_params = 3

    def membership_degree(self, value, a, b, c=None):
        if value is None or a is None or b is None or c is None:
            raise ValueError("Gaussian three function needs exact three arg \
                uments, less where given!")
        if a == 0:
            raise ValueError('Parameter a was 0 at MF mem degree')

        tmp1 = (value-c) / a
        denom = 1.0 + (tmp1**2.0)**b
        return 1.0 / denom

    def partial(self, value, var, a, b, c=None):
        result = 0

        if value is None or a is None or b is None or c is None:
            raise ValueError("Gaussian three function needs exact three arg \
                uments, less where given!")
        if a == 0:
            raise ValueError('Parameter a was 0 in MF deriv')

        tmp1 = (value-c) / a
        tmp2 = (tmp1**2) ** b
        denom = (1+tmp2) ** 2
        if var == 'a':
            result = 2.0*b*tmp2 / (a*denom)
        elif var == 'b':
            result = (-tmp2 * math.log(tmp1**2)) / denom
        elif var == 'c':
            result = 2.0*b * (value-c) * tmp1**(2.0*b - 2.0)
            result /= denom * a**2.0
        return result


class BellTwo(MembershipFunction):
    """ This class represents a Bell Shaped function with two parameters """

    def __init__(self):
        self.parameters = ['a', 'b']
        self.qtd_params = 2

    def membership_degree(self, value, a, b, c=None):
        arg = - ((value-b) / a)**2
        return max(math.exp(arg), MIN_MEMBERSHIP)

    def partial(self, value, var, a, b, c=None):
        result = 0
        denom = 1.0
        k = ((value-b) / a)**2
        if var == 'a':
            result = 2*(value-b)**2 * math.exp(-k)
            denom = a ** 3
        elif var == 'b':
            result = 2*(value-b) * math.exp(-k)
            denom = a ** 2
        else:
            raise ValueError('BellTwo has no parameter \'{}\''.format(var))
        return result / denom


class PiecewiseLogit(MembershipFunction):
    """ A piecewise linear approximation of logit (inverse of sigmoid) function
    with two parameters (p, q).
    """

    def __init__(self):
        self.parameters = ['p', 'q']
        self.qtd_params = 2

    def membership_degree(self, value, p, q, c=None):
        mem_degree = 0
        if value <= 0:
            mem_degree = p
        elif value > 0 and value < 1:
            mem_degree = (q-p)*value + p
        elif value >= 1:
            mem_degree = q
        return mem_degree

    def partial(self, value, var, p, q, c=None):
        if var in self.parameters:
            if var == self.parameters[0]:
                return self._partial_p(value)
            else:
                return self._partial_q(value)
        else:
            raise ValueError('piecewise logit invalid parameter ', var)

    def _partial_p(self, value):
        if value >= 1:
            return 0
        elif value < 1 and value > 0:
            return 1.0 - value
        else:
            return 1

    def _partial_q(self, value):
        return clip(value, 1, 0)

    def coefs(self, value, weight):
        return np.array([self.slope, self.indep, 0.0])
