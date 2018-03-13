
"""
example of calculations Lp norm for vector and matrix
"""

import numpy as np

def L1_norm(vec):
    return np.sum(np.abs(vec))

def L2_norm(vec):
    return np.sum(np.abs(vec) ** 2) ** 0.5

def Linf_norm(vec):
    return np.max(vec)

def Lp_norm(vec, p=2):
    if p < 1:
        raise Exception('p should be more or equal 1. {} given.'.format(p))
    return np.sum(np.abs(vec) ** p) ** (1 / p)

def Frobenius_norm(matrix):
    return np.sum(np.abs(matrix) ** 2) ** 0.5

if __name__ in '__main__':
    VECTOR = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print('vector    : {}'.format(VECTOR))
    print('L1 norm   : {:.2f}'.format(L1_norm(VECTOR)))
    print('L2 norm   : {:.2f}'.format(L2_norm(VECTOR)))
    for p in range(3, 6):
        print('L{} norm   : {:.2f}'.format(p, Lp_norm(VECTOR, p)))
    print('Linf norm : {:.2f}'.format(Linf_norm(VECTOR)))

    MATRIX = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print('')
    print('matrix: {}'.format(MATRIX))
    print('Frobenius norm: {:.2f}'.format(Frobenius_norm(MATRIX)))
