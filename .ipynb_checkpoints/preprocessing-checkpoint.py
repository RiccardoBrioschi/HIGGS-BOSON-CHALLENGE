""" Some functions to compute preprocessing procedures on data """

import numpy as np

def managing_missing_values(tx,features,threshold=0.5):
    """
    Filling np.nan with the median of the columns
    Arguments:
    Input:
    tx: matrix having missing values

    Output:
    tx: matrix after first processing
    """
    nan_per_columns = np.sum(np.isnan(tx),axis = 0)
    
    valid_columns = nan_per_columns <=threshold*tx.shape[0]
    
    # Drop features if less than 50% of rows have missing values
    
    features = features[valid_columns]
    tx=tx[:,valid_columns]

    for col in range(tx.shape[1]):
        median = np.nanmedian(tx[:,col])
        index = np.isnan(tx[:,col])
        tx[index,col] = median

    return tx, features,valid_columns

def capping_outliers(tx):

    """
    Capping outliers
    """
    for col in range(tx.shape[1]):
        indx1 = tx[:,col] > np.percentile(tx[:,col],95)
        indx2 = tx[:,col] < np.percentile(tx[:,col],5)
        tx[indx1,col]=np.percentile(tx[:,col],95)
        tx[indx2,col]=np.percentile(tx[:,col],5)
    return tx

def trigonometrics(tx,columns,features):
    """
    Define trigonometric function of angles variables.
    Arguments:
    tx :
    columns :
    features :
    
    Returns
    tx :
    features :
    """
    
    sin = np.sin(tx[:,columns])
    cos = np.cos(tx[:,columns])

    tx = np.hstack((tx,sin))
    tx = np.hstack((tx,cos))

    for col in columns:
        name_s='sin_' + features[col]
        name_c='cos_' + features[col]
        features = np.append(features,name_s)
        features = np.append(features,name_c)
        
    tx = np.delete(tx,columns,axis = 1)

    return tx, features

def log_transform(tx):
    """ 
    Function to transform skewed distributions
    Arguments
    tx :
    
    Returns
    tx :
    """
    tx = np.log(1+tx)
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
    
    mean = np.mean(data,axis = 0)
    std_data = data - mean
    std = np.std(std_data,axis = 0)
    std_data = std_data / std
    return std_data, mean, std

def build_poly(x, degree, no_interaction_factors_columns):
    
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree.
    After performing polynomial expansion, it also inserts interaction factors among 1 degree columns.
    It automatically adds an offset column.
    
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.
        no_interactions_factors_columns : integer, it is the number of columns related to trigonometric values that will
                                  not multiply with any column.
        
    Returns:
        poly: numpy array of shape (N,d+1)
    """
    
    len_without_offset_and_expansion = x.shape[1]-no_interaction_factors_columns
    
    phi=np.ones((x.shape[0],1))
    
    for i in range(1,degree+1):
        phi=np.c_[phi,x**i]

    # we now introduce interaction factors (we start from column index one to avoid multiplying columns for the offset)
    
    for i in range(1,len_without_offset_and_expansion-1):
        for j in range(i+1,len_without_offset_and_expansion):
            phi = np.c_[phi, phi[:,i]*phi[:,j]]
            
    return phi

    
    
    
    
    

