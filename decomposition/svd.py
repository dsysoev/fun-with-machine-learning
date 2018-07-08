
"""
Implementation of Singular Value Decomposition (SVD)
based on this post https://jeremykun.com/2016/05/16/singular-value-decomposition-part-2-theorem-proof-algorithm/
"""

import numpy as np


def L2_norm(vec):
    """Calculate L2 norm of vertor

    returns: float
    """
    return np.sum(np.abs(vec) ** 2) ** 0.5

def random_unit_vector(size):
    """Create normalized random unit vector

    size: lenght of vector

    returns: list
    """
    # create vector based on normal distribution
    unnormalized = np.random.normal(0, 1, size=size)
    # calculate L2 norm of vector
    norm = L2_norm(unnormalized)
    # normalized vector
    return unnormalized / norm

def svd_1d(A, epsilon=1e-10):
    """The one-dimensional SVD

    A: matrix
    epsilon: tolerance

    Returns:

    VT - right singular vector
    """
    n, m = A.shape
    # create random normalized unit vector
    lastV, currentV = None, random_unit_vector(m)
    # calculate A.T * A
    # A.T * A is a rectangular matrix
    ATA = np.dot(A.T, A)

    while True:
        lastV = currentV
        currentV = np.dot(ATA, lastV)
        currentV = currentV / L2_norm(currentV)
        if abs(np.dot(currentV, lastV)) > 1 - epsilon:
            return currentV

def svd(A, epsilon=1e-10, random_state=None):
    """Compute the singular-value decomposition of a matrix.

    A: matrix
    epsilon: tolerance
    random_state: int

    Returns: U, S, VT
    """

    np.random.seed(random_state)
    # shape of matrix
    # m - number of columns in matrix
    n, m = A.shape
    svdSoFar = []

    for i in range(m):
        # loop for each column
        # create temporary matrix as copy of original
        # and substact singular values from it
        # for all previuos columns
        matrixFor1D = A.copy()
        for singular, u, v in svdSoFar[:i]:
            # calculate product of two vectors
            matrix = np.outer(u, v)
            # substact previous singular values
            matrixFor1D -= singular * matrix
        # calculate svd for remaining matrix
        v = svd_1d(matrixFor1D, epsilon=epsilon)
        # next singular vector unnormalized
        u_unnormalized = np.dot(A, v)
        # get L2 norm of vertor
        sigma = L2_norm(u_unnormalized)
        # calculate next singular value
        u = u_unnormalized / sigma
        # append
        svdSoFar.append((sigma, u, v))
    # transform it into matrices of the right shape
    singularValues, us, vs = [np.array(x) for x in zip(*svdSoFar)]

    return us.T, singularValues, vs


if __name__ == "__main__":

    data = np.array([
        [2, 5, 3],
        [1, 2, 1],
        [4, 1, 1],
        [3, 5, 2],
        [5, 3, 1],
        [4, 5, 5],
        [2, 4, 2],
        [2, 2, 5],
    ], dtype='float64')

    # calculate mean vector
    data_mean = data.mean(axis=0)
    # extract mean vector from original matrix
    data_centered = data - data_mean
    # performe SVD
    # X' = U * S * VT
    # X - n-by-m matrix
    #
    # U - n-by-n matrix left singular vectors
    # S - singular values
    # VT - m-by-m matrix right singular vectors
    U, S, VT = svd(data_centered, epsilon=1e-10, random_state=37)
    # calculate restored matrix based on SVD
    restored = np.dot(U, np.dot(np.diag(S), VT)) + data_mean
    # calculate delta between original matrix and restored
    is_equal = (np.round(data - restored) == 0).all()
    print('data:\n{}'.format(data))
    print('U:\n{}'.format(U))
    print('S:\n{}'.format(np.diag(S)))
    print('VT:\n{}'.format(VT))
    print('\nOriginal matrix and restores are equal: {}'.format(is_equal))
