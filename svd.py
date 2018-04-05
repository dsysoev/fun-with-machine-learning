
"""
Implementation of Singular Value Decomposition (SVD)
based on this post https://jeremykun.com/2016/05/16/singular-value-decomposition-part-2-theorem-proof-algorithm/
"""

import numpy as np


def L2_norm(vec):
    """
    calculate L2 norm of vertor

    returns: float
    """
    return np.sum(np.abs(vec) ** 2) ** 0.5

def random_unit_vector(size):
    """
    create normalized random unit vector

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
    """
    The one-dimensional SVD
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

def svd(A, epsilon=1e-10):
    """
    Compute the singular-value decomposition of a matrix.

    Returns: U, singularValues, V
    """
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
    ratings = np.array([
        [2, 5, 3],
        [1, 2, 1],
        [4, 1, 1],
        [3, 5, 2],
        [5, 3, 1],
        [4, 5, 5],
        [2, 4, 2],
        [2, 2, 5],
    ], dtype='float64')
    # performe SVD
    U, singvalues, V = svd(ratings, epsilon=1e-10)
    # calculate restored matrix based on SVD
    restored = np.dot(U, np.dot(np.diag(singvalues), V))
    # calculate delta between original matrix and restored
    delta = np.round(ratings - restored)
    # print(U, singvalues, V)
    print(delta)
