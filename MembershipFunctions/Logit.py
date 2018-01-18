from LinguisticLabel import LinguisticLabel


class Logit(LinguisticLabel):
    """ A piecewise linear approximation of logit (inverse of sigmoid) function
    with two parameters (p, q).
    """
    def __init__(self):
        LinguisticLabel.__init__(self)
        self.__low = 0.001
        self.__high = 1

    def membership_degree(self, value, params):
        memDegree = 0
        if value <= self.__low:
            memDegree = self.__low
        elif self.__low < value and self.__high > value:
            a = (params[1] - params[0]) / (self.__high - self.__low)
            b = self.__low - a * value
            memDegree = a * value + b
        elif value >= self.__high:
            memDegree = self.__high
        return memDegree

    def derivative_at(self, value, var):
        pass
