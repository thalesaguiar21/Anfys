from __future__ import print_function
from fuzzy import mem_funcs as mfs
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

qtd_inputs = len(data[0][0][0])

smp = 1
for dt in data:
    con_fun = mfs.PiecewiseLogit()
    network = TsukamotoModel(QTD_MFS, INP_N, con_fun, mfs.BellTwo())
    network.learn_hybrid_online(dt, smp, max_epochs=300, tsk=filename)
    smp += 1
