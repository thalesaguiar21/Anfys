from MembershipFunctions.GaussianFuzzySubset import GaussianFuzzySubset
from MembershipFunctions.AbsoluteFuzzySubset import AbsoluteFuzzySubset
# from InferenceFunctions.CentroidStrategy import CentroidStrategy
from math import sqrt


value = 0.3
gauss = GaussianFuzzySubset(begin=1, end=2, mean=0, dev=sqrt(0.2))
print('Created a {0}'.format(gauss.name))
absolute = AbsoluteFuzzySubset(begin=-0.2, end=0.8)
print('Created a {0}'.format(absolute.name))

v = absolute.membershipValue(value)
print('The membership value for {0} is {1}'.format(value, v))
