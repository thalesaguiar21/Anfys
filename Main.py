from __future__ import print_function
from fuzzy import mem_funcs as mfs
from speech.anfis import TsukamotoModel
from data import utils as dutils
# import pdb
# from sklearn.datasets.samples_generator import make_blobs

# filename = str(raw_input('Enter a input folder: '))

QTD_MFS = 2
SET_NAMES = {'fsew0_4_': 4,
             'fsew0_5_': 5,
             'msak0_4_': 4,
             'msak0_5_': 5}
FILE_SETS = ['fsew0_4_', 'msak0_4_', 'fsew0_5_', 'msak0_5_']


for fset in FILE_SETS:
    for nrun in range(15):
        data = []
        INP_N = SET_NAMES[fset]
        for i in range(1, 461):
            fname = fset + dutils.next_file(i, 3) + '.txt'
            data.append(dutils.get_pairs(fname))

        qtd_inputs = len(data[0][0][0])

        smp = 1
        for dt in data:
            con_fun = mfs.PiecewiseLogit()
            network = TsukamotoModel(QTD_MFS, INP_N, con_fun, mfs.BellTwo())
            network.learn_hybrid_online(
                dt, smp, tol=1e-5, max_epochs=500, tsk=fset, r=nrun
            )
            smp += 1
