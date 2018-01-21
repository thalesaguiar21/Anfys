from MembershipFunctions import FuzzySubset
from MembershipFunctions import GaussianThree
from MembershipFunctions import Logit
from SpeechUtils import Alphabet
from random import random
from Anfis import Anfis


INPUT_SIZE = 6
LABELS = 39
CON_PARAMS_SIZE = 2
PRE_PARAMS_SIZE = 3

# Reading PHN -> VALUE file
phn_list = []
phn_values = []
phn_file = open('phoneme_relative_frequency.txt').read()
file_rows = phn_file.split('\n')

for row in file_rows:
    row = row.split('\t')
    phn_list.append(row[0])
    phn_values.append(float(row[1]))

alph = Alphabet(phn_list, phn_values)

inputs = [
    ([random() for i in range(INPUT_SIZE)], random()) for i in range(1)
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
    [0 for i in range(CON_PARAMS_SIZE)] for j in range(LABELS)
]

anfis = Anfis(precedents, consequents, consParams)
anfis.train_by_hybrid_online(6, 1e-8, inputs)
