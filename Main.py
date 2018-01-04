from MembershipFunctions.FuzzySubset import FuzzySubset
from MembershipFunctions.GaussianTwo import GaussianTwo
from MembershipFunctions.DiscreteSigmoid import DiscreteSigmoid
from InferenceFunctions.CentroidStrategy import CentroidStrategy
from random import random
from Anfis import Anfis

inputs = [random() for i in range(2)]

precedents = [
    FuzzySubset(
        [GaussianTwo() for i in range(2)],
        [[random() for i in range(3)] for i in range(2)]
    ) for i in range(2)
]

consequents = [
    DiscreteSigmoid() for i in range(2)
]

consParams = [
    [random() for i in range(2)] for j in range(2)
]
print 'Consequent params:\n'
print consParams
print ''

anfis = Anfis(precedents, consequents, CentroidStrategy(), 0.02)
anfis.consParams = consParams

anfis.forwardPass(inputs)
