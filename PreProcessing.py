import numpy as np
import numpy.ma as ma


def PCA(X):
        """
        Assumes observations in X are passed as rows of a numpy array.
        X must be standardized!!!!
        """
        N=X.shape[0]
        D=X.shape[1]
        
        C=(1/N)*X.T.dot(X)
        
        # Calculate the eigenvalues and eigenvectors of the covariance matrix
        e_values, e_vectors = np.linalg.eigh(C)

        # Sort eigenvalues and their eigenvectors in descending order
        e_ind_order = np.flip(e_values.argsort())
        e_values = e_values[e_ind_order]
        e_vectors = e_vectors[e_ind_order]

        # Save the first n_components eigenvectors as principal components
        principal_components = np.take(e_vectors, np.arange(D), axis=0)


        return np.matmul(X, principal_components.T)

def managing_missing_values(X):
    """
    change the np.nan with the median of the columns
    """
    np.where(np.isnan(X), ma.array(X, mask=np.isnan(X)).mean(axis=0), X)

    return X








