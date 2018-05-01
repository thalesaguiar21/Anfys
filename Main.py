from random import random
from fuzzy.mem_funcs import BellTwo, PiecewiseLogit
from speech.anfis import TsukamotoModel

"""
# Reading input file
data_file = open('data.txt').read()
data_rows = data_file.split('\n')

INPUT_SIZE = int(data_rows[0])
LABELS = int(data_rows[1])
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
    output = tmp_row[INPUT_SIZE:(INPUT_SIZE + (LABELS ** INPUT_SIZE))]
    data.append((entry, output))
if data[-1] == ([], []):
    del data[-1]
"""
data = [([2, 3, 2], 5),
        ([0.5, 1, 3], 4)]

sets_size = [3, 3]
prec_params = [[random() + 1, random() + 1] for i in xrange(sum(sets_size))]

prec_fun = BellTwo()
con_fun = PiecewiseLogit()
network = TsukamotoModel(sets_size, prec_params, prec_fun, con_fun)
network.learn_hybrid_online(data, max_epochs=400, prod=False)
