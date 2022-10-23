'''
Some  functions for dataset division wrt the categorical variable
'''
import numpy as np

def divide_indices_in_subsets(tx,categorical_column):
    """
    Function that returns boolean mask to divide dataset in categories according to PRI_num_jet variable
    """
    
    indices_cat_0 = tx[:,categorical_column] == np.float(0)
    indices_cat_1 = tx[:,categorical_column] == np.float(1)
    indices_cat_2_3 = (tx[:,categorical_column] == np.float(2)) | (tx[:,categorical_column] == np.float(3))
    
    return indices_cat_0, indices_cat_1, indices_cat_2_3

def divide_dataset_in_subsets(tx,y,ids,indices):
    """
    Function that actually divides the dataset according to the boolean masks passed as arguments
    """
    
    return tx[indices],y[indices],ids[indices]

def return_right_prediction(prediction, ids):
    prediction_col = np.column_stack([prediction])
    ids = np.column_stack([ids])

    ind_pred = np.hstack((ids,prediction_col))
    ind_pred = ind_pred[ind_pred[:, 0].argsort()]

    return ind_pred[:,1:].ravel(), ind_pred[:,:1].ravel()
