# import numpy as np
from sklearn.cluster import KMeans
# import pdb


def compute_centroids(filename1, filename2):
    data1 = open(filename1, 'r')
    data2 = open(filename2, 'r')
    datalines = data1.readlines()
    datalines.extend(data2.readlines())

    phonemes = []
    inputs = []
    for l in datalines:
        values = l.strip('\n').split('\t')
        if values[-1] not in phonemes:
            phonemes.append(values[-1])
        inputs.append([float(i) for i in values[:-1]])

    print 'Phonemes:\t{}\nDatasize:\t{}'.format(len(phonemes), len(inputs))

    n_phonemes = len(phonemes)
    kmeans = KMeans(n_clusters=n_phonemes,
                    random_state=0
                    ).fit(inputs)

    sample_centers = kmeans.predict(inputs)
    print 'Centers: ' + str(len(sample_centers))

    fileprefix = filename1.split('_')[1]
    with open('c_' + fileprefix + '.txt', 'w') as file:
        n_line = 0
        for line, center in zip(datalines, sample_centers):
            values = line.strip('\n').split('\t')
            values[-1] = max(kmeans.cluster_centers_[center])
            values = [float(v) for v in values]
            qtd_values = len(values)
            str_lin = ('{:20}\t' * qtd_values).format(*values)
            file.write(str_lin + '\n')
            n_line += 1

    print 'Saved results into {}'.format('c_' + fileprefix + '.txt')


def get_pairs(filename):
    my_dir = 'C:\\Users\\thalesaguiar\\Documents'
    my_dir += '\\Dev\\Python\\ANFIS\\datafiles\\'
    data = open(my_dir + filename, 'r')
    datalines = data.readlines()
    d_pair_tmp = []
    for line in datalines:
        d_pair_tmp.append(
            [
                float(i.strip(' '))
                for i in line.strip('\n').strip('\t').split('\t')
            ]
        )

    d_pair = []
    for i in xrange(len(d_pair_tmp)):
        inputs = [float(m) for m in d_pair_tmp[i][:-1]]
        output = float(d_pair_tmp[i][-1])
        d_pair.append((inputs, output))
    data.close()
    return d_pair
