from random import random
from fuzzy.mem_funcs import BellThree, PiecewiseLogit
from speech.anfis import TsukamotoModel
from datafiles import data_utils as dutils
# from sklearn.datasets.samples_generator import make_blobs

qtd_mfs = 2
filename = 'c_5F.txt'
data = dutils.get_pairs(filename)
qtd_inputs = len(data[0][0])


# Separate the data into 1000 samples each
start = 0
end = 1000
data_splitted = []
data_size = len(data)
while start < data_size:
    data_splitted.append(data[start:end])
    start = end
    end += 1000

# For each 1000 samples compute the outcomes
samp = 1
for dt in data_splitted:
    print '=' * 50
    print '{}\t{}th 1000 samples'.format(filename, samp)
    sets_size = [qtd_mfs for i in range(qtd_inputs)]
    prec_params = [
        [random() + 0.5, random() + 0.5, random() + 0.5]
        for i in xrange(sum(sets_size))
    ]

    prec_fun = BellThree()
    con_fun = PiecewiseLogit()
    network = TsukamotoModel(sets_size, prec_params, prec_fun, con_fun)
    network.learn_hybrid_online(dt, max_epochs=200, prod=True, setnum=samp)
    print '=' * 50
    samp += 1
