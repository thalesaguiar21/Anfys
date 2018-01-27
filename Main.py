from MembershipFunctions import FuzzySubset
from MembershipFunctions import GaussianThree
from MembershipFunctions import Logit
from random import random
from Anfis import Anfis


# Reading input file
data_file = open('data.txt').read()
data_rows = data_file.split('\n')

INPUT_SIZE = int(data_rows[0])
LABELS = int(data_rows[1])
CON_PARAMS_SIZE = 2
PRE_PARAMS_SIZE = 3

data = []

for line in range(len(data_rows) - 2):
    line = line + 2
    tmp_row = data_rows[line].split('\t')
    if tmp_row[-1] == '':
        del tmp_row[-1]
    for col in range(len(tmp_row)):
        tmp_row[col] = float(tmp_row[col])
    entry = tmp_row[:INPUT_SIZE]
    output = tmp_row[INPUT_SIZE:]
    data.append((entry, output))
if data[-1] == ([], []):
    del data[-1]

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
anfis.train_by_hybrid_online(2, 1e-8, data[:1])
