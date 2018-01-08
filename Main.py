from MembershipFunctions import FuzzySubset
from MembershipFunctions import GaussianTwo
from MembershipFunctions import DiscreteSigmoid
from InferenceFunctions import CentroidStrategy
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

anfis = Anfis(precedents, consequents, CentroidStrategy())
anfis.consParams = consParams

rs, out_vec = anfis.forwardPass(inputs)
print 'Inferred value is {} for the output vector: {}'.format(rs, out_vec)
