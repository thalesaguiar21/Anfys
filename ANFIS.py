from MembershipFunctions.GaussianFuzzySubset import GaussianFuzzySubset
from math import sqrt


value = 1.5
gauss = GaussianFuzzySubset(begin=1, end=2, mean=0, dev=sqrt(0.2))
print('Created a Gaussian Fuzzy Subset')
res = gauss.membershipValue(value)
print('The membership value for {0} is {1}'.format(value, res))
