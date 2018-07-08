
"""
Implementation of Principal component analysis (PCA)
https://en.wikipedia.org/wiki/Principal_component_analysis
"""

import numpy as np
from svd import svd


class PCA(object):
    """Principal component analysis (PCA)

    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space.

    """
    def __init__(self, n_components=None, random_state=None, epsilon=1e-10):
        self.n_components = n_components
        self.random_state = random_state
        self.epsilon = epsilon
        self.U, self.S, self.VT = None, None, None
        self.X_mean = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        """Fit the model to the data X.

        Parameters:
        X: original matrix for fit

        """
        if self.n_components is None:
            # set maximum number of components
            self.n_components = X.shape[1]
        # calculate mean vector
        self.X_mean = X.mean(0)
        # extract mean vector from matrix
        X_centered = X - self.X_mean
        # performe SVD
        self.U, self.S, self.VT = svd(
            X_centered, epsilon=self.epsilon, random_state=self.random_state)
        n_samples = X.shape[0]
        # calculate explained variance
        self.explained_variance_ = (self.S ** 2) / (n_samples - 1)
        self.explained_variance_ = self.explained_variance_[:self.n_components]
        # calculate explained variance ratio
        total_var = self.explained_variance_.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components]

    def transform(self, X):
        """Apply dimensionality reduction on X.

        Parameters:
        X : matrix for projection

        Returns:

        X_new: projected matrix
        """
        X_centered = X - self.X_mean
        X_transform = np.dot(X_centered, self.VT[:self.n_components].T)
        return X_transform

    def inverse_transform(self, X):
        """Transform data back to its original space.

        Parameters:
        X : projected matrix

        Returns:
        X_original : Original matrix

        """
        X_original = np.dot(X, self.VT[:self.n_components]) + self.X_mean
        return X_original


if __name__ == "__main__":

    from sklearn.decomposition import PCA as PCA_

    data = np.array([
        [2, 5, 3],
        [1, 2, 1],
        [4, 1, 1],
        [3, 5, 2],
        [5, 3, 1],
        [4, 5, 5],
        [2, 4, 2],
        [2, 2, 5],
    ], dtype='int64')

    n_components = 2
    pca = PCA(n_components=n_components, random_state=37)
    pca.fit(data)

    pca_ = PCA_(n_components=n_components, random_state=37)
    pca_.fit(data)

    check_transform = (np.round(
        pca.transform(data) - pca_.transform(data), decimals=3) == 0).all()

    X_transform = pca.transform(data)
    check_inverse = (np.round(
        pca.inverse_transform(X_transform) - pca_.inverse_transform(X_transform),
        decimals=3) == 0).all()

    print('data:\n{}'.format(data))
    print('Data transformed:\n{}'.format(X_transform))
    print('Is transform correct: {}'.format(check_transform))
    print('Inverse transformed data:\n{}'.format(pca.inverse_transform(X_transform)))
    print('Is inverse transform correct: {}'.format(check_inverse))
