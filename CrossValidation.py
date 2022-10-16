import numpy as np
from implementations import reg_logistic_regression
from costs import *

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.
    
    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_log(y, x, k_indices, k, lambda_, gamma, initial_w, max_iters):
    """return the loss of ridge regression for a fold corresponding to k_indices
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527597114, 0.33555914361295175)
    """
    
    ind1 = np.concatenate([k_indices[0:k,:].ravel(),k_indices[k+1:,:].ravel()])
    
    x_tr = x[ind1]
    x_test = x[k_indices[k]]
    
    y_tr = y[ind1]
    y_test = y[k_indices[k]]
 
    w, loss_tr = reg_logistic_regression(y_tr, x_tr, lambda_ ,initial_w, max_iters, gamma)

    loss_te=compute_logloss_logistic_regression(y_test,x_test,w)

    return loss_tr, loss_te



def cross_validation_demo(y, x, k_fold, lambdas, gamma, initial_w, max_iters):
    """cross validation over regularisation parameter lambda.
    
    Args:
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """
    
    seed = 12
    k_fold = k_fold
    lambdas = lambdas
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # ***************************************************
    for lambda_ in lambdas:
        l_tr = 0
        l_te = 0       
        for n in range(k_fold):
            loss_tr, loss_te = cross_validation_log(y, x, k_indices, n, lambda_, gamma, initial_w, max_iters)
            l_tr += loss_tr
            l_te += loss_te
        l_tr = l_tr/k_fold
        l_te = l_te/k_fold
        rmse_tr.append(np.sqrt(2*l_tr))
        rmse_te.append(np.sqrt(2*l_te))
        
    # ***************************************************    

    best_rmse = min(rmse_te)
    best_lambda=lambdas[rmse_te.index(best_rmse)]
    
    return best_lambda, best_rmse


