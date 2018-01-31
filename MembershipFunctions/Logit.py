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
        memDegree = 0
        a = (params[1] - params[0]) / (self.__high - self.__low)
        if value <= self.__low:
            memDegree = params[0]
        elif self.__low < value and self.__high > value:
            memDegree = a * value
        elif value >= self.__high:
            memDegree = params[1]
        return memDegree

    def derivative_at(self, value, var, params):
        return (params[1] - params[0]) / (self.__high - self.__low)
