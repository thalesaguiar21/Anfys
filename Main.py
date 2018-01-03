from MembershipFunctions.GaussianTwo import GaussianTwo
from random import random


value = random()
premiseParams = [random() for i in range(3)]
gaussTwo = GaussianTwo()

rs = gaussTwo.membershipValue(value, premiseParams)
print premiseParams
print 'g({0}) = {1}'.format(value, rs)
