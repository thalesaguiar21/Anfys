import numpy as np
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


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
    # pdb.set_trace()
    n_phonemes = len(phonemes)
    kmeans = KMeans(n_clusters=n_phonemes,
                    random_state=0
                    ).fit(inputs)

    print 'Predicting...'
    sample_centers = kmeans.predict(inputs)
    print 'Phonemes:\t{}'.format(len(phonemes))
    print 'Datasize:\t{}'.format(len(inputs))
    print 'Centers: ' + str(len(kmeans.cluster_centers_))
    return (kmeans, np.array(inputs), filename1, sample_centers, datalines)


def write_to_file(filename1, filename2=None):
    rs = compute_kmeans(filename1, filename2)
    kmeans = rs[0]
    # X = rs[1]
    # f1 = rs[2]
    scenters = rs[3]
    datalines = rs[4]
    fileprefix = filename1.split('_')[1]
    with open('c_' + fileprefix + '.txt', 'w') as file:
        n_line = 0
        for line, center in zip(datalines, scenters):
            values = line.strip('\n').split('\t')
            values[-1] = kmeans.cluster_centers_[center][0]
            values = [float(v) for v in values]
            qtd_values = len(values)
            str_lin = ('{:20}\t' * qtd_values).format(*values)
            file.write(str_lin + '\n')
            n_line += 1

    print 'Saved results into {}'.format('c_' + fileprefix + '.txt')


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

    # fileprefix = filename1.split('_')[1]
    # with open('c_' + fileprefix + '.txt', 'w') as file:
    #     n_line = 0
    #     for line, center in zip(datalines, sample_centers):
    #         values = line.strip('\n').split('\t')
    #         values[-1] = max(kmeans.cluster_centers_[center])
    #         values = [float(v) for v in values]
    #         qtd_values = len(values)
    #         str_lin = ('{:20}\t' * qtd_values).format(*values)
    #         file.write(str_lin + '\n')
    #         n_line += 1

    # print 'Saved results into {}'.format('c_' + fileprefix + '.txt')


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
    # plt.title('Iris Clusters and Centroids')
    plt.xlabel('Power')
    plt.ylabel('Time(s)')
    plt.legend()

    plt.show()


def plot_kmeans_3D(filename1, filename2=None, qtd_points=200):
    rs = compute_kmeans(filename1, filename2)
    # kmeans = rs[0]
    X = rs[1]
    # f1 = rs[2]
    # scenters = rs[3]
    # datalines = rs[4]
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


def write_all():
    write_to_file('fsew0_4F_results.txt', 'msak0_4F_results.txt')
    write_to_file('fsew0_5F_results.txt', 'msak0_5F_results.txt')
    write_to_file('fsew0_6F_results.txt', 'msak0_6F_results.txt')
    write_to_file('fsew0_7F_results.txt', 'msak0_7F_results.txt')
