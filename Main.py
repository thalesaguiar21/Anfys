from __future__ import print_function
from random import random
from fuzzy.mem_funcs import BellTwo, PiecewiseLogit
from speech.anfis import TsukamotoModel
from data import utils as dutils
# import pdb
# from sklearn.datasets.samples_generator import make_blobs

filename = str(raw_input('Enter a input folder: '))

QTD_MFS = 3
SET_NAMES = {'F4': ('fsew0_4_', 4),
             'F5': ('fsew0_5_', 5),
             'M4': ('msak0_4_', 4),
             'M5': ('msak0_5_', 5)}
INP_N = SET_NAMES[filename][1]

data = []
for i in range(1, 461):
    fset = SET_NAMES[filename][0]
    fname = fset + dutils.next_file(i, 3) + '.txt'
    data.append(dutils.get_pairs(fname))

# pdb.set_trace()
qtd_inputs = len(data[0][0][0])

# For each 1000 samples compute the outcomes
samp = 1
print('{} {} {}'.format('=' * 5, filename, '=' * 5))
for dt in data:
    prec_fun = BellTwo()
    con_fun = PiecewiseLogit()
    network = TsukamotoModel(QTD_MFS, INP_N, con_fun)  # BellTwo
    network.learn_hybrid_online(dt, max_epochs=500, setnum=samp)
    samp += 1
