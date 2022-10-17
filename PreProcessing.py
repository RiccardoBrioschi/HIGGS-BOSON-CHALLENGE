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


        return e_values, principal_components, np.matmul(X, principal_components.T)

def managing_missing_values(tx):
    """
    change the np.nan with the median of the columns
    """
    nan_per_columns = np.sum(np.isnan(tx),axis = 0)

    tx=tx[:,nan_per_columns <= 0.5*tx.shape[0]]  #drop delle features con nan oltre 50%

    for col in range(tx.shape[1]):
        median = np.nanmedian(tx[:,col])
        index = np.isnan(tx[:,col])
        tx[index,col] = median

    return tx


def reject_outliers(y,tx_train,m):

    A=np.array([abs(tx_train[:,0] - np.mean(tx_train[:,0])) < m * np.std(tx_train[:,0])])
    
    for i in range(1,tx_train.shape[1]):
        a=np.array([abs(tx_train[:,i] - np.mean(tx_train[:,i])) < m * np.std(tx_train[:,i])])
        A = np.vstack((A,a))
        
    A=A.T

    mask = np.sum(A,axis=1)>15
    tx=tx_train[mask , :]
    y = y[mask]
    return y,tx


def capping_outliers(tx):

    """
    capping outliers
    """

    for col in range(tx.shape[1]):
        indx1 = tx[:,col] > np.percentile(tx[:,col],95)
        indx2 = tx[:,col] < np.percentile(tx[:,col],5)
        tx[indx1,col]=np.percentile(tx[:,col],95)
        tx[indx2,col]=np.percentile(tx[:,col],5)

    return tx


