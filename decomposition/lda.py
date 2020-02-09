
import numpy as np


class LDA(object):
    
    def __init__(self, n_components):
        self.n_components = n_components
    
    def _within_class_scatter_matrices(self, X, y):

        shape = X.shape[1]
        matrix = np.zeros((shape, shape))

        for i in np.unique(y):
            matrix += np.cov(X[y == i].T)

        return matrix
    
    def _between_class_scatter_matrix(self, X, y):
    
        mean_vectors = []
        for i in np.unique(y):
            mean_vectors.append(np.mean(X[y == i], axis=0))

        overall_mean = np.mean(X, axis=0)

        shape = X.shape[1]
        matrix = np.zeros((shape, shape))
        for i, mean_vec in enumerate(mean_vectors):  
            n = X[y == i + 1,:].shape[0]
            mean_vec = mean_vec.reshape(shape, 1)
            overall_mean = overall_mean.reshape(shape, 1)
            matrix += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

        return matrix
    
    def _create_eig_values_and_vectors(self, X, y):
        
        wc = self._within_class_scatter_matrices(X, y)
        bc = self._between_class_scatter_matrix(X, y)
        
        A = np.linalg.pinv(wc).dot(bc)

        eig_vals, eig_vecs = np.linalg.eig(A)
        
        return eig_vals, eig_vecs
    
    def fit(self, X, y):
        
        eig_vals, eig_vecs = self._create_eig_values_and_vectors(X, y)
        
        # Make a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
        
        shape = X.shape[1]
        self.W = np.hstack([eig_pairs[comp][1].reshape(shape, 1) for comp in range(self.n_components)])
        
        return self
    
    def transform(self, X, y):
        return X.dot(self.W).real
    
    def fit_transform(self, X, y):
        self = self.fit(X, y)
        return self.transform(X, y)

    
