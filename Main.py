from random import random
from fuzzy.mem_funcs import BellThree, PiecewiseLogit
from speech.anfis import TsukamotoModel
from sklearn.datasets.samples_generator import make_blobs
from numpy import array
import pdb

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
# data = make_blobs(n_samples=30, centers=3, n_features=3)

# data = [(inp.tolist(), rs) for inp, rs in zip(data[0], data[1])]

# with open('blobs.txt', 'w') as blobs:
#     blobs.write(str(data))
data =[([-4.144321636298676, 4.873850296693019, 0.28181350664655547], 1), ([6.275299551790036, -10.434007146996999, -1.5781645642582676], 0), ([8.734555326783129, -8.32090383914321, 0.9937728081183312], 0), ([6.606714547960099, -9.10830609721715, -1.7486744507374883], 0), ([6.4428459005932535, -10.901862855575843, -0.7418672607500552], 0), ([-2.9579064830501487, 4.918967785521174, 2.6604901545803705], 1), ([-7.12498132124833, 5.337606659583699, 2.695434517153929], 1), ([7.689746429937304, -9.822183033476094, -1.5267381922347538], 0), ([1.844276034685228, 2.5969763517947575, -2.527950570227177], 2), ([-4.949697672220885, 5.126543557459716, 2.133452777238545], 1), ([7.251620582572302, -9.942560892411551, -0.8572401510589056], 0), ([-5.636436505998632, 5.455180447098408, 1.2887204080949843], 1), ([9.262139736745418, -9.920013140666608, -1.8648438851917972], 0), ([3.7497841550024473, 1.9160279092836916, -1.3996846102751064], 2), ([-4.149171099234046, 7.352202978786044, 2.017847748353857], 1), ([9.565037840970042, -9.871183712349682, -2.384661665575494], 0), ([-4.332793897804452, 7.74802268650158, 0.8211455619864174], 1), ([2.353801012612986, -0.19552432461395797, -3.4376298293139644], 2), ([-4.025340876556751, 7.150564918463051, 0.7840087642589797], 1), ([1.3525472995778363, 0.8601162213350008, -2.9873935242042524], 2), ([8.526295068289349, -9.438196733860886, 1.9356091894491656], 0), ([3.3515211435803325, 0.27655956717893515, -1.676309902305371], 2), ([3.2725383639170915, 0.5735846523990236, -3.1554284180734107], 2), ([3.9491501953481087, 1.6036026049766399, -1.9453207574636076], 2), ([-2.8376295066040313, 4.594347518540185, 1.366918976855528], 1), ([2.1356819456498775, 2.9548694166140907, -1.900048493169614], 2), ([4.218510687688664, 2.079488367808392, -2.06576109512655], 2), ([8.673318290890071, -8.212403370972062, -0.7486099300291815], 0), ([-6.090828168713113, 6.1672060642628, 1.5527288368928103], 1), ([2.565443744648949, 1.6440951797904608, -2.1912911007515112], 2)]
sets_size = [3, 3, 3]

prec_params = [
    [random() + 0.5, random() + 0.5, random() + 0.5]
    for i in xrange(sum(sets_size))
]

# prec_params = [[0.346952378321, 1.25196582828],
#                [0.573537269257, 0.530943776613],
#                [1.44892407175, 1.52957482676],
#                [1.68775768827, 0.724328090663],
#                [0.256003297664, 0.845368142917],
#                [1.66150488962, 0.729725713452],
#                [0.367586732585, 0.210448756794],
#                [0.575638941796, 0.0550909332716],
#                [1.60538113778, 1.88591105533]]

prec_fun = BellThree()
con_fun = PiecewiseLogit()
network = TsukamotoModel(sets_size, prec_params, prec_fun, con_fun)
network.learn_hybrid_online(data, max_epochs=200, prod=True)
