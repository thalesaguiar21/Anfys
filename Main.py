from MembershipFunctions import FuzzySubset
from MembershipFunctions import GaussianTwo
from MembershipFunctions import DiscreteSigmoid
from InferenceFunctions import CentroidStrategy
from SpeechUtils import phoneme
from random import random, randint
from Anfis import Anfis
from numpy import array


INPUT_SIZE = 6
LABELS = 39
CON_PARAMS_SIZE = 2
PRE_PARAMS_SIZE = 3

inputs = [random() for i in range(INPUT_SIZE)]

precedents = [
    FuzzySubset(
        [GaussianTwo() for i in range(LABELS)],
        [[random() for i in range(PRE_PARAMS_SIZE)] for i in range(LABELS)]
    ) for i in range(INPUT_SIZE)
]

consequents = [
    DiscreteSigmoid() for i in range(LABELS)
]

consParams = [
    [random() for i in range(CON_PARAMS_SIZE)] for j in range(LABELS)
]
# print('Consequent paramaters are \n{}'.format(array(consParams)))

anfis = Anfis(precedents, consequents, CentroidStrategy())
anfis.consParams = consParams

rs, out_vec = anfis.forward_pass(inputs, phoneme(randint(0, 39)))
anfis.backward_pass(out_vec, [0] * 10, 1e-10)
'''print 'Inferred value is {} for the output vector: \n{}'.format(
    rs,
    array(out_vec)
)'''
