from MembershipFunctions.GaussianFuzzySubset import GaussianFuzzySubset
from InferenceFunctions.CentroidStrategy import CentroidStrategy
from random import random
from Anfis import Anfis

precedents = []
for i in range(2):
    precedents.append([GaussianFuzzySubset() for j in range(2)])

consequents = []
for i in range(2):
    consequents.append([random() for i in range(3)])

inference = CentroidStrategy()

anfis = Anfis(precedents, 2, consequents, inference)
anfis.fowardPass([random() for i in range(2)])
print 'Success!'
