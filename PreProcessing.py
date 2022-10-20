import numpy as np
import numpy.ma as ma

def managing_missing_values(tx,threshold=0.5):
    """
    Filling np.nan with the median of the columns
    Arguments:
    Input:
    tx: matrix having missing values

    Output:
    tx: matrix after first processing
    """
    nan_per_columns = np.sum(np.isnan(tx),axis = 0)

    # Drop features if less than 50% of rows have missing values
    
    tx=tx[:,nan_per_columns <= threshold*tx.shape[0]]  

    for col in range(tx.shape[1]):
        median = np.nanmedian(tx[:,col])
        index = np.isnan(tx[:,col])
        tx[index,col] = median

    return tx


def reject_outliers(y,tx_train,m):

    result=np.array([abs(tx_train[:,0] - np.mean(tx_train[:,0])) < m * np.std(tx_train[:,0])])
    
    for i in range(1,tx_train.shape[1]):
        col_i=np.array([abs(tx_train[:,i] - np.mean(tx_train[:,i])) < m * np.std(tx_train[:,i])])
        result= np.vstack((result,col_i))
        
    result = result.T

    mask = np.sum(result,axis=1)>15
    tx=tx_train[mask , :]
    y = y[mask]
    return y,tx


def capping_outliers(tx):

    """
    Capping outliers without modifying columns of categorical values.
    """
    for col in range(tx.shape[1]-4):
        indx1 = tx[:,col] > np.percentile(tx[:,col],95)
        indx2 = tx[:,col] < np.percentile(tx[:,col],5)
        tx[indx1,col]=np.percentile(tx[:,col],95)
        tx[indx2,col]=np.percentile(tx[:,col],5)
    return tx

def categorical_values(tx,column,N):

    """
    Function to handle categorical values labelled as PRI_jet_num
    """
    rows = tx.shape[0]
    for n in range(N):
        new_column = np.zeros((rows,1))
        index = tx[:,column]==np.float(n)
        new_column[index] = 1
        tx = np.hstack([tx,new_column])
    # We finally delete the column having the old values
    tx = np.delete(tx,column,axis = 1)
    return tx

def standardize(data):
    """ 
    This function standardizes the feature matrix.
    Returns:
    std_data : standardize data
    mean : mean of data
    std : standard deviation of data.
    """
    # The dataset has already been processed, so there are not nan values. Using np.nanmean or np.nanstd
    # is therefore not necessary.
    # We do not want to standardize the columns related to categorical values.
    
    float_data = data[:,:-4]
    mean = np.mean(float_data,axis = 0)
    std_data = float_data - mean
    std = np.std(std_data,axis = 0)
    std_data = std_data / std
    data[:,:-4]=std_data
    return data, mean, std


def build_model_data(y, X_without_offset):
    """
    Form (y,tx) to get regression data in matrix form (adding offset column).
    """
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), X_without_offset]
    return y, tx
