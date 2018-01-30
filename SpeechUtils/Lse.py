from numpy import eye, dot, array, zeros
initial_gamma = 10. ** 15


def lse(coefMatrix, rsMatrix, lamb=0.9, gamma=initial_gamma):
    """ Computes the Least Square Estimation for the matrices AX = B

    Parameters
    ----------
    coefMatrix : 2D list of double
        The coefficient matrix, or A
    rsMatrix : 2D list of double
        The result matrix, or B
    lamb : double
        An real value between 0 and 1. Defaults to 0.2
    gamma : double
        Initial values for main diag of S. Default to 10 ** 10

    Returns
    -------
    X : numpy array
        An approximation of X, the unknown vector
    """
    # pdb.set_trace()
    A = array(coefMatrix)
    S = eye(len(coefMatrix[0])) * gamma
    X = zeros((A.shape[1], 1))  # Change for ANFIS with more than 1 output
    B = array(rsMatrix)
    for i in range(len(A[:, 0])):
        a_i = array([A[i, :]])
        S = S - (
            array(
                dot(S, dot(a_i.T, dot(a_i, S))) /
                (lamb + dot(a_i, dot(S, a_i.T)))
            )
        )
        S = dot(S, 1.0 / lamb)
        X = X + (dot(S, dot(a_i.T, (B[i] - dot(a_i, X)))))
    return X


# _A = [
#     [
#         2.69210173e-45, -2.69210173e-45, 1.48643457e-45, -1.48643457e-45,
#         2.06102222e-45, -2.06102222e-45, 1.13798622e-45, -1.13798622e-45,
#         6.34986860e-08, -6.34986860e-08, 3.50605776e-08, -3.50605776e-08,
#         4.86133941e-08, -4.86133941e-08, 2.68417157e-08, -2.68417157e-08
#     ]
# ]

# _B = [[25]]

# result = lse(_A, _B)
# print result
# total = 0
# for i in range(16):
#     total += result[i] * _A[0][i]
# print total
