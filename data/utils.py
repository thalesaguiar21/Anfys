import numpy as np
import pdb
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('../')

or_folder = 'original'


def read_phonemes(fname):
    phonemes = []
    with open(fname, 'r') as file:
        lines = file.readlines()
        values = [v.split('\t') for v in lines]
        for l in values:
            phn = l[-1].strip('\t').strip('\n')
            if phn not in phonemes:
                phonemes.append(phn)
    return phonemes


def get_phns_from_run():
    my_dir = 'C:\\Users\\thalesaguiar\\Documents'
    my_dir += '\\Dev\\Python\\ANFIS\\data\\fsew0_4\\'
    mdict = {}
    indexes = 0
    pindex = []
    for i in xrange(1, 461):
        prompt = []
        with open(my_dir + 'fsew0_' + next_file(i, 3) + '_ext.txt') as f:
            lines = f.readlines()
            values = [v.split('\t') for v in lines]
            for val in values:
                prompt.append(val[-1].strip('\t').strip('\n'))
                indexes += 1
        mdict[i] = prompt
        pindex.append(indexes)
    my_dir = 'C:\\Users\\thalesaguiar\\Documents'
    my_dir += '\\Dev\\Python\\ANFIS\\results\\fsew0_4_H_4IN_2MF_13RNS.txt'
    with open(my_dir, 'r') as f:
        lines = f.readlines()
        runs = []
        start, end = [0 for _ in range(2)]
        for i in range(len(lines) / pindex[-1]):
            end = start + pindex[-1]
            testing = []
            phonemes = []
            # Get the 12454 phonemes
            for line in lines[start:end]:
                phonemes.append(
                    float(
                        line.strip('\n').strip('\t').split('\t')[-1].strip('\n').strip('\t')
                    )
                )
            # Join the phonemes into the prompts
            st = 0
            jotinha = 1
            for jay in pindex:
                testing.append(phonemes[st:jay])
                st = jay
                if len(testing[-1]) != len(mdict[jotinha]):
                    pdb.set_trace()
                jotinha += 1
            start = end
            runs.append(testing)
    return runs


def get_expected_phns():
    my_dir = 'C:\\Users\\thalesaguiar\\Documents'
    my_dir += '\\Dev\\Python\\ANFIS\\data\\fsew0_4\\'
    prompts = {}
    for i in xrange(1, 461):
        prompt = []
        with open(my_dir + 'fsew0_' + next_file(i, 3) + '_ext.txt') as f:
            lines = f.readlines()
            values = [v.split('\t') for v in lines]
            for val in values:
                prompt.append(val[-1].strip('\t').strip('\n'))
        prompts[i] = prompt
    return prompts


def read_centers(fname):
    centers = []
    for i in range(1, 461):
        with open('clustered/' + fname + next_file(i, 3) + '.txt') as file:
            lines = file.readlines()
            values = [v.strip('\n').strip('\t').split('\t') for v in lines]
            for val in values:
                c = float(val[-1].strip('\t').strip('\n'))
                if c not in centers:
                    centers.append(c)
    return centers


def compute_kmeans(filename1, filename2=None):
    print 'Reading file(s)...'
    data1 = open(filename1, 'r')
    datalines = data1.readlines()
    data1.close()
    if filename2 is not None:
        data2 = open(filename2, 'r')
        datalines.extend(data2.readlines())
        data2.close()

    phonemes = []
    inputs = []
    for l in datalines:
        values = l.strip('\n').split('\t')
        if values[-1] not in phonemes:
            phonemes.append(values[-1])
        inputs.append([float(i) for i in values[:-1]])

    print 'Clustering...'
    n_phonemes = len(phonemes)
    kmeans = KMeans(n_clusters=n_phonemes,
                    random_state=0
                    ).fit(inputs)
    return (kmeans, datalines)


def write_to_file(filename1, filename2=None):
    rs = compute_kmeans(filename1, filename2)
    kmeans = rs[0]
    datalines = rs[1]

    rsfolder = 'clustered'
    if not os.path.exists(rsfolder):
        os.makedirs(rsfolder)

    fileprefix = filename1.split('F')[0]
    for i in range(1, 461):
        fnum = next_file(i, 3)
        fname = fileprefix.split('_')[0] + '_' + fnum + '_ext.txt'
        extfile = open(fileprefix + '/' + fname, 'r')
        extfile_lines = extfile.readlines()
        inputs = []
        for line in extfile_lines:
            inputs.append(
                [
                    float(v.strip('\n'))
                    for v in line.strip('\n').split('\t')[:-1]
                ]
            )
        extfile.close()

        scenters = kmeans.predict(inputs)
        rs_fpath = rsfolder + '/' + fileprefix + '_' + str(fnum) + '.txt'
        with open(rs_fpath, 'w+') as file:
            for line, center in zip(datalines, scenters):
                values = line.strip('\n').split('\t')
                values[-1] = kmeans.cluster_centers_[center][0]
                values = [float(v) for v in values]
                qtd_values = len(values)
                str_lin = ('{:20}\t' * qtd_values).format(*values)
                file.write(str_lin + '\n')

        print 'Saved results into {}'.format(rs_fpath)


def extract_numerals(num):
    div = 1
    numerals = []
    while div > 0:
        div = num / 10
        numeral = num - div * 10
        if div == 0:
            numeral = num
        numerals.insert(0, str(numeral))
        num = div
    return numerals


def next_file(index, max_length):
    numerals = extract_numerals(index)
    result = ''
    for num in numerals:
        result += str(num)
    if len(result) < max_length:
        prefix = '0' * (max_length - len(result))
        result = prefix + result
    return result


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

    plot_kmeans_3D(kmeans, np.array(inputs))


def plot_kmeans(filename1, filename2=None, qtd_points=200):
    rs = compute_kmeans(filename1, filename2)
    means = rs[0]
    X = rs[1]
    print 'Plotting {} points in 2D...'.format(qtd_points)
    colors = means.cluster_centers_
    plt.scatter(X[:qtd_points, 0], X[:qtd_points, 1], s=300, c=colors)
    plt.scatter(
        means.cluster_centers_[:, 0],
        means.cluster_centers_[:, 1],
        s=300,
        c='red',
        label='Centroids'
    )
    plt.xlabel('Power')
    plt.ylabel('Time(s)')
    plt.legend()

    plt.show()


def plot_kmeans_3D(filename1, filename2=None, qtd_points=200):
    rs = compute_kmeans(filename1, filename2)
    X = rs[1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = X[:qtd_points, 0]
    y = X[:qtd_points, 1]
    z = X[:qtd_points, 2]

    ax.scatter(x, y, z, c='r', marker='o')
    ax.set_xlabel('xaxis')
    ax.set_ylabel('yaxis')
    ax.set_zlabel('zaxis')

    plt.show()


def get_pairs(filename):
    my_dir = 'C:\\Users\\thalesaguiar\\Documents'
    my_dir += '\\Dev\\Python\\ANFIS\\data\\clustered\\'
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


def write_all():
    write_to_file('fsew0_4F_results.txt')
    write_to_file('fsew0_5F_results.txt')
    write_to_file('msak0_4F_results.txt')
    write_to_file('msak0_5F_results.txt')


def compute_statistics(rs_matrix, ep=1, se=3, t=4):
    """ Compute the statistics for the given matrix. The mean
    squared error, mean of epochs and the mean of time

    Parameters
    ----------
    rs_matrix : 2D matrix
        The matrix with the results
    ep : int, defaults to 1
        The col with the number of epochs for each sample
    se : int, defaults to 3
        The column with the squared error for each sample
    t : int, defaults to 4
        The column with the total time for each sample
    """
    n_samples = len(rs_matrix)
    total_epochs, t_sqr_err, t_time = [0 for _ in range(3)]

    for samp in range(n_samples):
        total_epochs += rs_matrix[samp][:ep][ep - 1]
        t_sqr_err += rs_matrix[samp][:se][se - 1]
        t_time += rs_matrix[samp][:t][t - 1]

    n_samples = float(n_samples)
    print 'MSE: ' + str(t_sqr_err / n_samples)
    print 'Mean epochs: ' + str(total_epochs / n_samples)
    print 'Mean time: ' + str(t_time / n_samples)
