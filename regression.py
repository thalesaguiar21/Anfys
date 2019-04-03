from numpy import array, eye, zeros, dot


def lse_online(coef_matrix, rs_matrix, lamb=0.9, gamma=1000):
    """ Computes the Least Square Estimation for the matrices AX = B

    Parameters
    ----------
    coef_matrix : 2D list of double
        The coefficient matrix, or A
    rs_matrix : 2D list of double
        The result matrix, or B
    lamb : double, defaults to 0.9
        An real value between 0 and 1.
    gamma : double, defaults to 1000
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
