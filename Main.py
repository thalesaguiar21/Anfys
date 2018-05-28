from __future__ import print_function
from random import random
from fuzzy.mem_funcs import BellTwo, PiecewiseLogit
from speech.anfis import TsukamotoModel
from data import utils as dutils
# import pdb
# from sklearn.datasets.samples_generator import make_blobs

qtd_mfs = 3
set_names = {'F4': 'fsew0_4_',
             'F5': 'fsew0_5_',
             'M4': 'msak0_4_',
             'M5': 'msak0_6_'}


filename = str(raw_input('Enter a input folder: '))
data = []
for i in range(1, 461):
    fset = set_names[filename]
    fname = fset + dutils.next_file(i, 3) + '.txt'
    data.append(dutils.get_pairs(fname))

# pdb.set_trace()
qtd_inputs = len(data[0][0][0])

# For each 1000 samples compute the outcomes
samp = 1
print('{} {} {}'.format('=' * 5, filename, '=' * 5))
for dt in data:
    sets_size = [qtd_mfs for i in range(qtd_inputs)]
    prec_params = [
        [random() + 0.5, random() + 0.5]
        for i in xrange(sum(sets_size))
    ]

    prec_fun = BellTwo()
    con_fun = PiecewiseLogit()
    network = TsukamotoModel(sets_size, prec_params, prec_fun, con_fun)
    network.learn_hybrid_online(dt, max_epochs=500, prod=True, setnum=samp)
    samp += 1
