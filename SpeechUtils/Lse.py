from numpy import eye, dot, array, zeros, matrix


def lse(coefMatrix, rsMatrix, gamma=0.002):
    """ Computes the Least Square Estimation for the matrices AX = B

    Parameters
    ----------
    coefMatrix : 2D list of double
        The coefficient matrix, or A
    rsMatrix : 2D list of double
        The result matrix, or B
    gamma : double
        An positive real value

    Returns
    -------
    X : numpy array
        An approximation of X, the unknown vector
    """
    A = array(coefMatrix)
    S = eye(len(coefMatrix))
    X = zeros(A.shape[1], 1)
    for i in range(len(A[:, 0])):
        a_i = A[i, :]
        b_i = array(rsMatrix[i])
        S = S - (
            array(
                dot(dot(dot(S, matrix(a_i).T), matrix(a_i)), S))) \
            / (gamma + (dot(dot(S, a_i), a_i)))
        S *= 1 / gamma
        X = X + (dot(
            S,
            dot(matrix(a_i).T, (matrix(b_i) - dot(matrix(a_i), X))))
        )
    return X
