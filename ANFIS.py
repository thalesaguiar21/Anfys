from MembershipFunctions.GaussianFuzzySubset import GaussianFuzzySubset
from InferenceFunctions.CentroidStrategy import CentroidStrategy
from math import sqrt


value = 1.5
gauss = GaussianFuzzySubset(begin=1, end=2, mean=0, dev=sqrt(0.2))
print('Created a Gaussian Fuzzy Subset')
res = gauss.membershipValue(value)
print('The membership value for {0} is {1}'.format(value, res))

inp = [1, 2, 3]
w = [0.3, 0.5, 0.6]
centroid = CentroidStrategy()
result = centroid.infer(inputs=inp, weights=w)
print('The result is {0} by Centroid Inference Strategy.'.format(result))

