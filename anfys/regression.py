import numpy as np


def lse_online(coef_matrix, rs_matrix, lamb=0.98, gamma=1000):
    """ Computes the Least Square Estimation for the matrices AX = B

    Parameters
    ----------
    coef_matrix : 2D list of double
        The coefficient matrix, or A
    rs_matrix : 2D list of double
        The result matrix, or B
    lamb : double, defaults to 0.98
        Forgetting factor
    gamma : double, defaults to 1000
        Initial confiability of the system

    Returns
    -------
    X : numpy array
        An approximation of X, the unknown vector
    """
    A = np.array(coef_matrix)
    S = np.eye(coef_matrix.shape[1]) * gamma
    X = np.zeros((A.shape[1], 1))  # Change for ANFIS with more than 1 output
    B = np.array(rs_matrix)
    for i in range(len(A[:, 0])):
        a_i = np.array([A[i, :]])
        S = S - (
            np.array(
                np.dot(S, np.dot(a_i.T, np.dot(a_i, S))) /
                (lamb + np.dot(a_i, np.dot(S, a_i.T)))
            )
        )
        S = S * (1.0 / lamb)
        X = X + (np.dot(S, np.dot(a_i.T, (B[i] - np.dot(a_i, X)))))
    return X


def lse(coef_matrix, rs_matrix, gamma=1000):
    """ Computes the Least Square Estimation for the matrices AX = B

    Parameters
    ----------
    coef_matrix : 2D list of double
        The coefficient matrix, or A
    rs_matrix : 2D list of double
        The result matrix, or B
    gamma : double, defaults to 1000
        Initial values for main diag of S.

    Returns
    -------
    X : numpy array
        An approximation of X, the unknown vector
    """
    A = np.array(coef_matrix)
    S = np.eye(len(coef_matrix[0])) * gamma
    X = np.zeros((A.shape[1], 1))  # Change for ANFIS with more than 1 output
    B = np.array(rs_matrix)
    for i in range(len(A[:, 0])):
        a_i = np.array([A[i, :]])
        S = S - (
            np.array(
                np.dot(S, np.dot(a_i.T, np.dot(a_i, S))) /
                (1 + np.dot(a_i, np.dot(S, a_i.T)))
            )
        )
        X = X + (np.dot(S, np.dot(a_i.T, (B[i] - np.dot(a_i, X)))))
    return X
