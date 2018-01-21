from numpy import eye, dot, array, zeros, matrix


def lse(coefMatrix, rsMatrix, lamb=0.2, gamma=1000):
    """ Computes the Least Square Estimation for the matrices AX = B

    Parameters
    ----------
    coefMatrix : 2D list of double
        The coefficient matrix, or A
    rsMatrix : 2D list of double
        The result matrix, or B
    lamb : double
        An real value between 0 and 1. Defaults to 0.2

    Returns
    -------
    X : numpy array
        An approximation of X, the unknown vector
    """
    A = matrix(coefMatrix)
    S = eye(len(coefMatrix)) * gamma
    X = zeros((A.shape[1], 1))  # Change for ANFIS with more than 1 output
    for i in range(len(A[:, 0])):
        a_i = A[i, :]
        b_i = array(rsMatrix)  # For Online Batch, the b_i will be a double
        S = S - (
            array(
                dot(S, dot(a_i.T, dot(a_i, S))) /
                (lamb + dot(a_i, dot(S, a_i.T)))
            )
        )
        S *= 1 / lamb
        X = X + (dot(S, dot(a_i.T, (b_i - dot(a_i, X)))))
    return X
