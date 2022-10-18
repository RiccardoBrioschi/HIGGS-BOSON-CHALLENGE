import numpy as np
from implementations import *
from costs import *
from plots import *
from helpers import *


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


def cross_validation(y, x, k_indices, k, lambda_, degree):
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

    num_row = y.shape[0]
    
    ind1 = np.concatenate([k_indices[0:k,:].ravel(),k_indices[k+1:,:].ravel()])
    
    x_tr = x[ind1]
    x_test = x[k_indices[k]]
    
    y_tr = y[ind1]
    y_test = y[k_indices[k]]

    phi = build_poly(x_tr, degree)
    phi2 = build_poly(x_test, degree)

    w_opt = ridge_regression(y_tr, phi, lambda_)

    mse_test = compute_mse(y_test, phi2, w_opt)
    mse_train = compute_mse(y_tr, phi, w_opt)
    loss_tr = np.sqrt(2*mse_train)
    loss_te = np.sqrt(2*mse_test)

    return loss_tr, loss_te



def cross_validation_demo(y, tx, degree, k_fold, lambdas):
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
    degree = degree
    k_fold = k_fold
    lambdas = lambdas
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # ***************************************************
    for j in lambdas:
        l_tr = 0
        l_te = 0       
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y, tx, k_indices, k, j, degree)
            l_tr += loss_tr
            l_te += loss_te
        l_tr = l_tr/k_fold
        l_te = l_te/k_fold
        rmse_tr.append(l_tr)
        rmse_te.append(l_te)
        
    # ***************************************************    
    
    ind = np.argmin(rmse_te)
    best_lambda = lambdas[ind]
    best_rmse = rmse_te[ind]
    
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    
    print("For polynomial expansion up to degree %.f, the choice of lambda which leads to the best test rmse is %.5f with a test rmse of %.3f" % (degree, best_lambda, best_rmse))
    return best_lambda, best_rmse



def best_degree_selection(y, tx, degrees, k_fold, lambdas, seed = 1):
    """cross validation over regularisation parameter lambda and degree.
    
    Args:
        degrees: shape = (d,), where d is the number of degrees to test 
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_degree : integer, value of the best degree
        best_lambda : scalar, value of the best lambda
        best_rmse : value of the rmse for the couple (best_degree, best_lambda)
        
    >>> best_degree_selection(np.arange(2,11), 4, np.logspace(-4, 0, 30))
    (7, 0.004520353656360241, 0.28957280566456634)
    """
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # cross validation over degrees and lambdas: TODO
    
    rmse_tr=np.ones((len(degrees),len(lambdas)))
    rmse_te=np.ones((len(degrees),len(lambdas)))
    
    for d in degrees:
        for j in lambdas:  
            l_tr = 0
            l_te = 0 
            for k in range(k_fold):
                loss_tr, loss_te = cross_validation(y, tx, k_indices, k, j, d)
                l_tr += loss_tr
                l_te += loss_te
            l_tr = l_tr/k_fold
            l_te = l_te/k_fold
            rmse_tr[np.where(degrees==d),np.where(lambdas==j)] = l_tr
            rmse_te[np.where(degrees==d),np.where(lambdas==j)] = l_te
        
    # ***************************************************   
    # ***************************************************    
    row,col = np.unravel_index(np.argmin(rmse_te),rmse_te.shape)
    return degrees[row], lambdas[col], rmse_te[row,col]

