import numpy as np
from sklearn.cluster import KMeans
import pdb


def compute_centroids(filename):
    data = open(filename, 'r')
    datalines = data.readlines()
    datainputs = {}
    for l in datalines:
        values = l.strip('\n').split('\t')
        phn = values[-1]
        if phn in datainputs:
            datainputs[phn].append(np.array(values[:-1]).astype(np.float))
        else:
            datainputs[phn] = [values[:-1]]

    n_phonemes = len(datainputs)
    for phn in datainputs:
        kmeans = KMeans(n_clusters=n_phonemes,
                        random_state=0
                        ).fit(datainputs[phn])
        print kmeans.cluster_centers_

    for phn in datainputs:
        qtddata = len(datainputs[phn])
        total = 0
        # pdb.set_trace()
        for data in datainputs[phn]:
            total += np.array(data).astype(np.float).sum() / qtddata
        datainputs[phn] = total / qtddata

    filename = filename.split('_r')[0]
    with open('centroids_' + filename, 'w') as file:
        for line in datalines:
            phn = line.strip('\n').split('\t')[-1]
            no_phn = line.replace(phn, str(datainputs[phn]))
            file.write(no_phn)


def get_pairs(filename):
    my_dir = 'C:\\Users\\thalesaguiar\\Documents\\Dev\\Python\\ANFIS\\datafiles\\'
    data = open(my_dir + filename, 'r')
    datalines = data.readlines()
    d_pair_tmp = []
    for line in datalines:
        d_pair_tmp.append(line.strip('\n').split('\t'))

    d_pair = []
    for i in xrange(len(d_pair_tmp)):
        inputs = [float(m) for m in d_pair_tmp[i][:-1]]
        output = float(d_pair_tmp[i][-1])
        d_pair.append((inputs, output))
    data.close()
    return d_pair
