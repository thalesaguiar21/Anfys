import sys
sys.path.append('../')
l_format = '{:<3}\t{:16.7f}\t{:16}\t{:16}\t{:16}\n'
f_header = '{:<3}\t{:<16}\t{:<16}\t{:<16}\n'.format(
    *['ep', 'k', 'err', 'time', 'phn']
)


def w(f, ep=0, k=0, err=0, time=0, phn=0.0, header=False):
    """ Write a formatted line into the given file.

    Parameters
    ----------
    f : file
        The file where to write
    ep : int, defaults to 0
        Number of epochs
    err : int, defaults to 0
        Total error
    time: int, defaults to 0
        Total time in s
    phn : float, defaults to 0.0
        The phoneme result
    header : boolean, defaults to False
        To print or not the header
    """
    if header:
        f.write(f_header)
    else:
        f.write(l_format.format(ep, k, err, time, phn))


def result_matrix(fname, phn=False):
    """ Read a column ('\t') separated data file from 'results/'
    into a matrix.

    Parameters
    ----------
    fname : string
        The result file
    phn : boolean, defaults to False
        Include or not the las row (phonemes)
    """
    with open('results/' + fname) as results:
        rsline = results.readlines()
        rsmatrix = []
        for line in rsline:
            line = line.split('\t') if phn else line.split('\t')[:-1]
            rsmatrix.append(
                [float(v.strip('\n').strip('\t')) for v in line]
            )
        return rsmatrix
