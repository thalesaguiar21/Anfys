from MembershipFunctions import FuzzySubset
from MembershipFunctions import GaussianThree
from MembershipFunctions import Logit
from SpeechUtils import Alphabet
from random import random, randint
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
    ([3, 4, 7, 6, 3, 3], 0.4842671974610999),
    ([8, 5, 2, 1, 1, 5], 0.8222039989717432)
    # ([7, 8, 0, 3, 7, 8], 0.08793257014525835),
    # ([3, 2, 1, 0, 3, 1], 0.11308970646896765),
    # ([2, 5, 4, 0, 0, 8], 0.3031476176449488),
    # ([5, 3, 2, 8, 5, 2], 0.09869895897667702),
    # ([1, 3, 8, 0, 6, 0], 0.020195775315741127),
    # ([2, 5, 0, 8, 0, 3], 0.13207790045980772),
    # ([8, 4, 5, 4, 5, 7], 0.49925306591050245),
    # ([3, 2, 0, 6, 8, 8], 0.36906736595085365)
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
anfis.train_by_hybrid_online(10, 1e-8, inputs)
