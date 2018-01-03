class FuzzySubset():
    """ Fuzzy subset
    """

    def __init__(self, labels, params):
        self.labels = labels
        self.params = params

    def evaluate(self, value):
        return [self.labels.membershipValue(value, par) for par in self.params]
