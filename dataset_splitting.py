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

def reordering_predictions(predictions,ids):
    """
    Helper function to reorder predictions given by each model before creating submission file
    Arguments:
    predictions : ndarray of predictions
    ids : non sorted ids corresponding to each prediction
    
    Returns:
    predictions : predictions sorted according to ids
    ids : sorted array of ids
    """
    
    new_row_order = np.argsort(ids)
    return predictions[new_row_order],ids[new_row_order]
