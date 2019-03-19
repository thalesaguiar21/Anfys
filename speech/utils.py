from __future__ import print_function
from __future__ import division
from numpy import eye, dot, array, zeros
import math
import pdb


def lse_online(coef_matrix, rs_matrix, lamb=0.9, gamma=10000):
    """ Computes the Least Square Estimation for the matrices AX = B

    Parameters
    ----------
    coef_matrix : 2D list of double
        The coefficient matrix, or A
    rs_matrix : 2D list of double
        The result matrix, or B
    lamb : double, defaults to 0.9
        An real value between 0 and 1.
    gamma : double, defaults to 10000
        Initial values for main diag of S.

    Returns
    -------
    X : numpy array
        An approximation of X, the unknown vector
    """
    A = array(coef_matrix)
    S = eye(coef_matrix.shape[1]) * gamma
    X = zeros((A.shape[1], 1))  # Change for ANFIS with more than 1 output
    B = array(rs_matrix)
    for i in range(len(A[:, 0])):
        a_i = array([A[i, :]])
        S = S - (
            array(
                dot(S, dot(a_i.T, dot(a_i, S))) /
                (lamb + dot(a_i, dot(S, a_i.T)))
            )
        )
        S = S * (1.0 / lamb)
        X = X + (dot(S, dot(a_i.T, (B[i] - dot(a_i, X)))))
    return X


def lse(coef_matrix, rs_matrix, gamma=10000):
    """ Computes the Least Square Estimation for the matrices AX = B

    Parameters
    ----------
    coef_matrix : 2D list of double
        The coefficient matrix, or A
    rs_matrix : 2D list of double
        The result matrix, or B
    gamma : double, defaults to 10000
        Initial values for main diag of S.

    Returns
    -------
    X : numpy array
        An approximation of X, the unknown vector
    """
    A = array(coef_matrix)
    S = eye(len(coef_matrix[0])) * gamma
    X = zeros((A.shape[1], 1))  # Change for ANFIS with more than 1 output
    B = array(rs_matrix)
    for i in range(len(A[:, 0])):
        a_i = array([A[i, :]])
        S = S - (
            array(
                dot(S, dot(a_i.T, dot(a_i, S))) /
                (1 + dot(a_i, dot(S, a_i.T)))
            )
        )
        X = X + (dot(S, dot(a_i.T, (B[i] - dot(a_i, X)))))
    return X


def p_progress(qtd_data, p, tsk, csymb='#', psymb='-', basis=10):
    """ Print a progress bar and refresh the line

    qtd_data : int
        Total number of samples
    p : int
        Current sample
    csymb : charactere, defaults to '#'
        Symbol to use when complete one basis
    psymb : charactere, defaults to '-'
        Symbol to be used on uncompleted basis
    basis : int, defaults to 10
        The bar size
    tsk : string
        The task for wich the pair belongs
    """
    todo = int((qtd_data - p) / qtd_data * basis)
    done = basis - todo
    pbar = '[{}]'.format((csymb * done) + (psymb * todo))
    print('{}\t{:3}/ {:<3}\t||\t{} /460'.format(
        pbar, p, qtd_data, tsk),
        end='\r'
    )


def get_phn(task, *values):
    """ Retrieve the phoneme is the range of the given value

    Parameters
    ----------
    values : list of values
        The values infered from the ANFIS

    Returns
    -------
    A list of strings with the phonemes (word)
    """
    phonemes = [
        'dh', 'i', 's', 'w', 'z', 'ii', 'iy', 'f', 'r', 'uh', 'oo',
        'ei', 'ou', 'th', 'v', 't', 'l', 'jh', 'uu', 'n', 'm', 'b', 'ai', 'k',
        'ng', 'h', 'aa', 'd', 'sh', 'o', 'e', 'y', 'eir', 'g', 'ow', 'a', 'u',
        'p', 'ch', 'zh', 'oi'
    ]
    return [phonemes[int(math.ceil(i * 10))] for i in values]


    # word = []
    # centers = []
    # if task == 'fsew0_4':
    #     centers = [
    #         0.084830616173, -0.120916721883, 0.0052523752367, 0.349618999285,
    #         0.0704499979749, -0.1224546941, 0.0318944333359, -0.0265193468704,
    #         0.148804612707, -0.209538645983, 0.107485866013, 0.364181707836,
    #         -0.171489554635, -0.0811546446374, 0.2128472792, 0.0263184705267,
    #         -0.0243306595346, -0.247921497248, 0.707536823847, 0.133176431287,
    #         -0.19751896072, -0.0439524343198, -0.132583264154, -0.374574850341,
    #         -0.30334100348, 0.260752134429, 0.218442939963, 0.359731842763,
    #         -0.0890305678404, 0.448918869641, 0.594956119786, 0.225278776214,
    #         0.399928538654, 0.0202980438131, -0.159382962584, -0.331437361398,
    #         -0.375033088697, 0.2514080061, 0.696238110606, 0.394713056006,
    #         0.669586355143
    #     ]
    # elif task == 'mask0_4':
    #     centers = [
    #         -0.0505427954385, 0.00546123227783, 0.221496076299, -0.22791828332,
    #         0.172838990416, -0.150471707084, 0.0927162840129, -0.0579789977991,
    #         -0.0757452140126, 0.0140344172041, -0.00502733749666,
    #         0.213470061599, 0.508397424308, -0.0503262431381, 0.335512294746,
    #         -0.0716950102343, -0.0580274849114, -0.123090563567,
    #         0.381903478354, 0.0531512880716, 0.202104923535, -0.0945996611028,
    #         -0.27399341553, -0.175385096079, 0.468950393596, 0.56286308992,
    #         0.0888352631904, 0.0263713017238, 0.144573896459, -0.0526381894848,
    #         -0.149325752556, 0.0888763488439, 0.371614574387, -0.11013072081,
    #         0.0550827948402, -0.144388718364, 0.586522298321, 0.0328286941248,
    #         -0.134937694601, 0.318263919867, -0.285804766172
    #     ]
    # elif task == 'fsew0_5':
    #     centers = [
    #         -0.0672335116707, -0.178820473826, -0.082580887353, 0.225734742028,
    #         0.0566297072609, -0.176058846824, 0.277302063129, 0.0901950784411,
    #         0.0394674884661, -0.369984461024, 0.237427153191, -0.106298424163,
    #         -0.0815602327301, 0.270934045595, 0.03184725327, -0.185714003958,
    #         -0.00215001252071, 0.310531898321, -0.12759464318, -0.038545828836,
    #         -0.225823471112, 0.536969772274, 0.0237630788366, -0.21848449864,
    #         -0.311841033448, -0.0749346279855, 0.308055753589, 0.308116761295,
    #         0.0757422632445, 0.186121382332, 0.493551372897, 0.301460631303,
    #         -0.175467431281, -0.0735130234718, -0.335524808525, 0.190600683968,
    #         0.729582043095, -0.303434333099, 0.768035037595, 0.664769050387,
    #         0.451585454342
    #     ]
    # elif task == 'mask0_5':
    #     centers = [
    #         -0.035504640483, -0.120967187689, 0.223987834369, -0.0304358113296,
    #         0.0238191215756, -0.100544148428, -0.0760810789963, 0.134518631324,
    #         0.239287324496, -0.0950657181681, 0.237026429802, -0.0905658643491,
    #         0.00742816601621, -0.103463995127, -0.0776896172833,
    #         0.0440807131878, -0.198979826688, 0.0944388814881,
    #         -0.0473525325189, -0.123445531165, 0.352548463543, -0.23720025695,
    #         0.478457415742, 0.0599424432488, 0.696781302088, 0.513808530827,
    #         -0.0104779208273, 0.253607361935, -0.173643304106, 0.104210074703,
    #         -0.149089545689, 0.250140251456, 0.168817220551, 0.430360111912,
    #         0.0125795684825, 0.00561546000559, -0.166746504668,
    #         -0.132419306996, -0.318702682364, -0.208409277834,
    #         0.470637923779
    #     ]

    # for val in values:
    #     m_index = 0
    #     m_aux = abs(centers[0] - val)
    #     for i in range(1, len(phonemes)):
    #         m_curr = abs(centers[i] - val)
    #         if m_curr < m_aux:
    #             m_aux = m_curr
    #             m_index = i
    #     word.append(phonemes[m_index])
    # return word


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
    dists = zeros((m, n))
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
