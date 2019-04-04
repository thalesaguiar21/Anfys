import numpy as np


def levenshtein_distance(expected, result, subc=1):
    """ Compute the levenshtein distance between the given strings

    ld_a_b(i, j) = if min(i, j) == 0, max(i, j)
                   otherwise, min(
                       ld_a_b(i - 1, j) + 1,
                       ld_a_b(i, j - 1) + 1,
                       ld_a_b(i - 1, j - 1) + sub_cost,
                   )
    where i = |a| and j = |b|.

    Parameters
    ----------
    expected : list
        A list with the correct symbols
    result : list
        A list with the resulted symbols
    subc : int, defaults to 1
        The cost of a substitution
    """
    # Matrix with costs
    e_tmp = expected[:]
    r_tmp = result[:]
    e_tmp.insert(0, ' ')
    r_tmp.insert(0, ' ')
    m = len(e_tmp)
    n = len(r_tmp)
    dists = np.zeros((m, n))
    sub_cost = 0

    for i in range(m):
        dists[i, 0] = i

    for j in range(n):
        dists[0, j] = j

    sub_cost = 0
    for j in range(1, n):
        for i in range(1, m):
            if e_tmp[i] == r_tmp[j]:
                sub_cost = 0
            else:
                sub_cost = subc
            dists[i, j] = min(
                dists[i - 1, j] + 1,
                dists[i, j - 1] + 1,
                dists[i - 1, j - 1] + sub_cost
            )
    return dists[m - 1, n - 1]
