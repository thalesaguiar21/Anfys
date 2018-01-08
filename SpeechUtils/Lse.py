def lse(coefMatrix, rsMatrix):
    """ Computes the Least Square Estimation for the matrices AX = B

    Parameters
    ----------
    coefMatrix : 2D list of double
        The coefficient matrix, or A
    rsMatrix : 2D list of double
        The result matrix, or B

    Returns
    -------
    X : list of double
        The unknown vector
    """
    pass


def T(matrix):
    """ Returns the tranpose of a matrix

    Parameters
    ----------
    matrix : list
        The original matrix

    Returns
    -------
    matrixT : list
        The transpose of matrix
    """
    if type(matrix[0]) is not []:
        return matrix
    else:
        matrixT = [
            [0 for j in range(len(matrix))] for i in range(len(matrix[0]))
        ]
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                matrixT[j][i] = matrix[i][j]
        return matrixT
