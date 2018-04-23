from random import random
from fuzzy.mem_funcs import BellTwo, PiecewiseLogit
from speech.anfis import TsukamotoModel
import pprint

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
data = [([0, 1], 1),
        ([0, 0], 0),
        ([1, 0], 1),
        ([1, 1], 0)]

sets_size = [3, 3]
prec_params = [
    [0.5810385287233912, 1.2868153860009148],
    [1.3473850681828206, 0.8389800640658862],
    [1.339900880732897, 0.09401431112042458],
    [1.5922668716131554, 1.0868876078022072],
    [1.8865873866422764, 0.22993741183044225],
    [1.34072223255989, 1.5755485241834029]]
# [1.47763039586, 1.29578608372],
# [1.32773919238, 0.119812931268],
# [1.32801874768, 0.15856952585],
# [1.32776144698, 0.962786935246]]
prec_fun = BellTwo()
con_fun = PiecewiseLogit()
network = TsukamotoModel(sets_size, prec_params, prec_fun, con_fun)
network.learn_hybrid_online(data, max_epochs=200)

"""
precedents = [
    FuzzySubset(
        LABELS,
        [[random() for i in range(PRE_PARAMS_SIZE)] for i in range(LABELS)],
        BellThree()
    ) for i in range(INPUT_SIZE)
]

consequents = [
    PiecewiseLogit() for i in range(LABELS ** INPUT_SIZE)
]

anfis = Anfis(precedents, consequents)
anfis.set_debug(False)
anfis.train_by_hybrid_online(5, 1e-8, data[:1]) """
