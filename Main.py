from __future__ import print_function
from fuzzy import mem_funcs as mfs
from speech.anfis import TsukamotoModel
from data import utils as dutils
import pdb
# from sklearn.datasets.samples_generator import make_blobs

prompts = dutils.get_expected_phns()

QTD_MFS = 2
SET_SIZE = 461
SET_NAMES = {'fsew0_4_': 4, 'fsew0_5_': 5, 'msak0_4_': 4, 'msak0_5_': 5}
FILE_SETS = ['fsew0_4_', 'msak0_4_', 'fsew0_5_', 'msak0_5_']

i = 1
for f in FILE_SETS:
    print('{} - {}'.format(i, f))
    i += 1

f_num = int(raw_input('Enter the data file: ')) - 1
n_runs = int(raw_input('Enter the number of times: '))
heurist = raw_input('Run with k-heuristc (y/n)?  ')
kh = True if heurist == 'y' else False
# Create result file name
INP_N = SET_NAMES[FILE_SETS[f_num]]
rs_fname = '{}{}_{}IN_{}MF_{}RNS.txt'.format(
    FILE_SETS[f_num], 'H' if kh else '', INP_N, QTD_MFS, n_runs
)

# Run the task for N times
for i in range(n_runs):
    data = []
    for i in range(1, 2):
        fname = FILE_SETS[f_num] + dutils.next_file(i, 3) + '.txt'
        data.append(dutils.get_pairs(fname))
    smp = 1
    for dt in data:
        con_fun = mfs.PiecewiseLogit()
        network = TsukamotoModel(QTD_MFS, INP_N, con_fun, mfs.BellTwo())
        network.learn_hybrid_online(
            dt, smp, prompts[smp], tsk=rs_fname, heurist=kh
        )
        smp += 1
