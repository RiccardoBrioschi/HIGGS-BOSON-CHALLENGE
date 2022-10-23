'''
Some helper functions for the DataSet Division wrt the categorical variable
'''
import numpy as np



def divide_dataset_in_subsets(tx, y, cat_v, categories = (0,1,2)):
    list_y = []
    list_tx = []
    
    for i in categories:
        if i == 2:
            
            aux = tx[np.where(tx[:,cat_v] >= i)]
            aux_y = y[np.where(tx[:,cat_v] >= i)]
        
        else:
            
            aux = tx[np.where(tx[:,cat_v] == i)]
            aux_y = y[np.where(tx[:,cat_v] == i)] 

        aux = np.delete(aux, cat_v, 1)
        list_y.append(aux_y)
        list_tx.append(aux)
    
    return list_tx, list_y


def divide_test_data_in_subset(tx_test, categories = (0,1,2)):
    list_tx_test = []
    
    for i in categories:
        if i == 2:

            aux = tx_test[tx_test[:,23] >= i]
        
        else:

            aux = tx_test[tx_test[:,23] == i]

        aux = np.delete(aux, 23, 1)
        list_tx_test.append(aux)
    
    return list_tx_test

def add_ids_column(tx, ids, features):
    ids_col = np.column_stack([ids])
    tx = np.hstack((ids_col, tx))
    features = np.insert(features, 0, 'IDs')

    return tx, features.tolist()

def ids_divided(list_matrix):
    ids_after_subset = []
    for i in range(len(list_matrix)):
        ids_after_subset = np.concatenate((ids_after_subset, list_matrix[i][:,:1].ravel()))
    
    return ids_after_subset

def delete_ids(list_matrix, features):
    for i in range(len(list_matrix)):
        list_matrix[i] = list_matrix[i][:,1:]

        return list_matrix, features[1:]

def add_ids_column_test(tx_test, test_ids):
    test_ids_col = np.column_stack([test_ids])
    tx_test = np.hstack((test_ids_col, tx_test))

    return tx_test

def delete_ids_test(list_matrix_test):
    for i in range(len(list_matrix_test)):
        list_matrix_test[i] = list_matrix_test[i][:,1:]

        return list_matrix_test

def return_right_prediction(prediction, ids):
    prediction_col = np.column_stack([prediction])
    ids = np.column_stack([ids])

    ind_pred = np.hstack((ids,prediction_col))
    ind_pred = ind_pred[ind_pred[:, 0].argsort()]

    return ind_pred[:,1:].ravel(), ind_pred[:,:1].ravel()