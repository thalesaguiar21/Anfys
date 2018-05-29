import sys
sys.path.append('../')
l_format = '{:<3}\t{:<16}\t{:<16}\t{:<16}\n'
f_header = ['ep', 'k', 'err', 'time']


def f_name(subset, anfis):
    fname = 'results/{}_{}INP_{}MF'
    fname = fname.format(subset, anfis.inp_n, anfis.mf_n * anfis.inp_n)
    return fname + '.txt'


def w(f, ep=0, k=0, err=0, time=0, header=False):
    if header:
        f.write(l_format.format(*f_header))
    else:
        f.write(l_format.format(ep, k, err, time))
