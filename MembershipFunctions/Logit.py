from LinguisticLabel import LinguisticLabel


class Logit(LinguisticLabel):
    """ A piecewise linear approximation of logit (inverse of sigmoid) function
    with two parameters (p, q).
    """
    def __init__(self):
        LinguisticLabel.__init__(self)
        self.__low = 1e-15
        self.__high = 0.99999

    def membership_degree(self, value, params):
        mem_degree = 0
        a = (params[1] - params[0]) / (self.__high - self.__low)
        if value <= self.__low:
            mem_degree = params[0]
        elif self.__low < value and self.__high > value:
            mem_degree = a * value
        elif value >= self.__high:
            mem_degree = params[1]
        return mem_degree

    def derivative_at(self, value, var, params):
        return (params[1] - params[0]) / (self.__high - self.__low)
