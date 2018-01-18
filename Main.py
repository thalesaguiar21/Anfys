from MembershipFunctions import FuzzySubset
from MembershipFunctions import GaussianThree
from MembershipFunctions import Logit
from InferenceFunctions import CentroidStrategy
from SpeechUtils import MappedAlphabet
from random import random, randint
from Anfis import Anfis


INPUT_SIZE = 6
LABELS = 39
CON_PARAMS_SIZE = 2
PRE_PARAMS_SIZE = 3


phonemeList = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER',
    'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z',
    'ZH'
]

phnNum = len(phonemeList)

thresholds = [0] * phnNum
for i in range(1, phnNum):
    thresholds[i] = thresholds[i - 1] + random()

alph = MappedAlphabet(phonemeList, thresholds)

inputs = [
    (
        [random() for i in range(INPUT_SIZE)],
        alph.symbol(randint(0, phnNum - 1))
    ) for i in range(10)
]

precedents = [
    FuzzySubset(
        [GaussianThree() for i in range(LABELS)],
        [[random() for i in range(PRE_PARAMS_SIZE)] for i in range(LABELS)]
    ) for i in range(INPUT_SIZE)
]

consequents = [
    Logit() for i in range(LABELS)
]

consParams = [
    [random() * 10 for i in range(CON_PARAMS_SIZE)] for j in range(LABELS)
]

for params in consParams:
    if params[0] == params[1]:
        params[0] += 0.1
# print('Consequent paramaters are \n{}'.format(array(consParams)))

anfis = Anfis(precedents, consequents, CentroidStrategy(), alph)
anfis.consParams = consParams

anfis.train_by_hybrid_online(6, 3e-3, inputs)
'''print 'Inferred value is {} for the output vector: \n{}'.format(
    rs,
    array(out_vec)
)'''
