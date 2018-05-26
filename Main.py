from random import random
from fuzzy.mem_funcs import BellTwo, PiecewiseLogit
from speech.anfis import TsukamotoModel
from datafiles import data_utils as dutils
# from sklearn.datasets.samples_generator import make_blobs

qtd_mfs = 3

data = dutils.get_pairs('centroids_fsew0_4F')
qtd_inputs = len(data[0][0])
sets_size = [qtd_mfs for i in range(qtd_inputs)]
prec_params = [
    [random() + 0.5, random() + 0.5]
    for i in xrange(sum(sets_size))
]

prec_fun = BellTwo()
con_fun = PiecewiseLogit()
network = TsukamotoModel(sets_size, prec_params, prec_fun, con_fun)
network.learn_hybrid_online(data, max_epochs=500, prod=True)
