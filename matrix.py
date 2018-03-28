
"""
Implementation of simple matrix class on pure python
"""

from numpy import array as Array

class Matrix(object):
    """ Implementation of matrix object based on numpy array """

    def __init__(self, lst):
        """ initializing matrix oject based on lists """
        self.matrix = lst

    def __str__(self):
        """ present matrix as string """
        lst = ["\t".join(map(str, line)) for line in self.matrix]
        return "\n".join(lst)

    def __mul__(self, other):
        """ multiplication matrix by scale
        return Matrix object

        >>> lst = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        >>> ((Matrix(lst) * 10) == Matrix(Array(lst) * 10)).all()
        True
        """
        n, m = self.shape
        C = [[0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                C[i][j] = self.matrix[i][j] * other
        return Matrix(C)

    def __add__(self, other):
        """ additional matrix by scale
        return Matrix object

        >>> lst = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        >>> ((Matrix(lst) + 10) == Matrix(Array(lst) + 10)).all()
        True
        """
        n, m = self.shape
        C = [[0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                C[i][j] = self.matrix[i][j] + other
        return Matrix(C)

    def __sub__(self, other):
        """ substraction matrix by scale
        return Matrix object

        >>> lst = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        >>> ((Matrix(lst) - 1) == Matrix(Array(lst) - 1)).all()
        True
        """
        n, m = self.shape
        C = [[0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                C[i][j] = self.matrix[i][j] - other
        return Matrix(C)

    def __truediv__(self, other):
        """
        division matrix by scale
        return Matrix object

        >>> lst = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        >>> ((Matrix(lst) / 2) == Matrix(Array(lst) / 2)).all()
        True
        """
        n, m = self.shape
        C = [[0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                C[i][j] = self.matrix[i][j] / other
        return Matrix(C)

    def __rtruediv__(self, other):
        """ right division matrix by scale
        return Matrix object

        >>> lst = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        >>> ((2 / Matrix(lst)) == Matrix(2 / Array(lst))).all()
        True
        """
        n, m = self.shape
        C = [[0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                C[i][j] = other / self.matrix[i][j]
        return Matrix(C)

    def __pow__(self, other):
        """ power matrix by scale
        return Matrix object

        >>> lst = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        >>> ((Matrix(lst) ** 2) == Matrix(Array(lst) ** 2)).all()
        True
        """
        n, m = self.shape
        C = [[0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                C[i][j] = self.matrix[i][j] ** other
        return Matrix(C)

    def __rpow__(self, other):
        """ right power scale by matrix
        return Matrix object

        >>> lst = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        >>> ((2 ** Matrix(lst)) == Matrix(2 ** Array(lst))).all()
        True
        """
        n, m = self.shape
        C = [[0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                C[i][j] = other ** self.matrix[i][j]
        return Matrix(C)

    def __matmul__(self, other):
        """ Implementation of naive square matrix multiplication
        return Matrix object

        >>> lst1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        >>> lst2 = [[1], [2], [3]]
        >>> ((Matrix(lst1) @ Matrix(lst2)) == Matrix(Array(lst1) @ Array(lst2))).all()
        True
        """
        if self.shape[1] != other.shape[0]:
            raise Exception('shape of matrices differ '
                            '{} {}'.format(self.shape, other.shape))
        n, m, p = self.shape[0], other.shape[1], other.shape[0]
        C = [[0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    C[i][j] += self.matrix[i][k] * other.matrix[k][j]
        return Matrix(C)

    def __rmul__(self, other):
        """ right multiplication

        >>> lst1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        >>> lst2 = [[1], [2], [3]]
        >>> ((Matrix(lst1) @ Matrix(lst2)) == Matrix(Array(lst1) @ Array(lst2))).all()
        True
        """
        return self * other

    def dot(self, other):
        """ return matrix multiplication

        >>> lst1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        >>> lst2 = [[1], [2], [3]]
        >>> ((Matrix(lst1).dot(Matrix(lst2))) == Matrix(Array(lst1).dot(Array(lst2)))).all()
        True
        """
        return self @ other

    def __radd__(self, other):
        """ right additional

        >>> lst = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        >>> ((10 + Matrix(lst)) == Matrix(10 + Array(lst))).all()
        True
        """
        return self + other

    def __rsub__(self, other):
        """ right substraction

        >>> lst = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        >>> ((2 - Matrix(lst)) == Matrix(2 - Array(lst))).all()
        True
        """
        return self * -1 + other

    def __eq__(self, other):
        """ implementation __eq__ call

        >>> lst = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        >>> (Matrix(lst) == Matrix(Array(lst))).all()
        True
        """
        if not isinstance(other, Matrix):
            raise NotImplementedError
        if self.shape != other.shape:
            raise Exception('shape of matrices differ {} {}'.format(
                                                    self.shape, other.shape))
        n, m = self.shape
        C = [[None for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                C[i][j] = self.matrix[i][j] == other.matrix[i][j]
        return Matrix(C)

    def all(self):
        """ return True if all elements in matrix is True

        >>> lst = [[True], [True]]
        >>> Matrix(lst).all()
        True

        >>> lst = [[True], [True], [False]]
        >>> Matrix(lst).all()
        False
        """
        n, m = self.shape
        for i in range(n):
            for j in range(m):
                if self.matrix[i][j] == False:
                    return False
        return True

    @property
    def shape(self):
        """ return shape of matrix

        >>> lst = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        >>> Matrix(lst).shape == Array(lst).shape
        True
        """
        return (len(self.matrix), len(self.matrix[0]))

    @property
    def T(self):
        """ return transposed matrix

        >>> lst = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        >>> (Matrix(lst).T == Matrix(Array(lst).T)).all()
        True
        """
        return Matrix(list(map(list, zip(*self.matrix))))


if __name__ in '__main__':
    lst = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    mx = Matrix(lst)

    lst1 = [[1], [2], [3]]
    mx1 = Matrix(lst1)
    print('Matrix:\n{}'.format(mx))
    mx_test = Array(lst)
