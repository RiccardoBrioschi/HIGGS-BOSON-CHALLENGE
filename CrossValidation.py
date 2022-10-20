import numpy as np
from implementations import *
from costs import *
from plots import cross_validation_visualization
from preprocessing import *

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.
    
    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_log(y, x, k_indices, k, lambda_, gamma,degree, max_iters):
    """
    Return the loss of ridge regression for a fold corresponding to k_indices
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)
    """

    num_row = y.shape[0]
    
    train_indices = np.concatenate([k_indices[:k,:].ravel(),k_indices[k+1:,:].ravel()])
    test_indices = k_indices[k]
    
    x_train = x[train_indices]
    x_test = x[test_indices]
    
    y_train = y[train_indices]
    y_test = y[test_indices]

    x_train_temp = x_train[:,:-4]
    x_test_temp = x_test[:,:-4]

    # We compute polynomial expansion and add the offset column
    
    poly_train = build_poly(x_train_temp, degree)
    poly_test = build_poly(x_test_temp, degree)

    # Adding last columns (categorical variables)
    
    poly_train =np.hstack((poly_train,x_train[:,-4:]))
    poly_test =np.hstack((poly_test,x_test[:,-4:]))
    
    initial_w = np.zeros(poly_train.shape[1])
    
    w_opt,_ = reg_logistic_regression(y_train,poly_train,lambda_,initial_w,max_iters,gamma)

    loss_tr = compute_logloss_logistic_regression(y_train,poly_train,w_opt)
    loss_te = compute_logloss_logistic_regression(y_test,poly_test,w_opt)

    return loss_tr, loss_te



def cross_validation_demo_log(y, tx, k_fold, lambdas, gamma, max_iters,degrees,seed = 10):
    """
    Cross validation over regularisation parameter lambda.
    
    Args:
        degrees: list of degrees of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """

    k_fold = k_fold
    lambdas = lambdas
    # Split data in k fold
    k_idx = build_k_indices(y, k_fold, seed)
    
    # Define matrices to store the loss of training data and test data
    
    rmse_tr = np.zeros((len(lambdas),len(degrees)))
    rmse_te = np.zeros((len(lambdas),len(degrees)))
    
    for i,param in enumerate(lambdas):
        for j,deg in enumerate(degrees):
            l_tr = 0
            l_te = 0
            for k in range(k_fold):
                loss_tr, loss_te = cross_validation_log(y,tx, k_idx,k,param, gamma,deg, max_iters)
                l_te += loss_te
                l_tr += loss_tr
            l_te = l_te/k_fold
            l_tr = l_tr/k_fold
            rmse_tr[i,j] = l_tr
            rmse_te[i,j] = l_te
    
    idx_lambda,idx_degree = np.unravel_index(np.argmin(rmse_te),rmse_te.shape)
    best_degree,best_lambda,best_rmse = degrees[idx_degree],lambdas[idx_lambda],rmse_te[idx_lambda,idx_degree]
    
    if len(degrees) == 1:
        cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    
    print("The choice of lambda which leads to the best test rmse is %.5f with a test rmse of %.3f. The best degree is %.1f" % (best_lambda, best_rmse,best_degree))
    return best_degree, best_lambda, best_rmse



def cross_validation_ridge(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression for a fold corresponding to k_indices
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)
    """

    num_row = y.shape[0]
    
    train_indices = np.concatenate([k_indices[:k,:].ravel(),k_indices[k+1:,:].ravel()])
    test_indices = k_indices[k]
    
    x_train = x[train_indices]
    x_test = x[test_indices]
    
    y_train = y[train_indices]
    y_test = y[test_indices]

    x_train_temp = x_train[:,:-4]
    x_test_temp = x_test[:,:-4]

    # We compute polynomial expansion and add the offset column
    
    poly_train = build_poly(x_train_temp, degree)
    poly_test = build_poly(x_test_temp, degree)

    # Adding last columns (categorical variables)
    
    poly_train =np.hstack((poly_train,x_train[:,-4:]))
    poly_test =np.hstack((poly_test,x_test[:,-4:]))

    w_opt, _ = ridge_regression(y_train,poly_train, lambda_)

    mse_train = compute_loss_linear_regression(y_train,poly_train, w_opt)
    mse_test = compute_loss_linear_regression(y_test,poly_test, w_opt)
    loss_tr = np.sqrt(2*mse_train)
    loss_te = np.sqrt(2*mse_test)

    return loss_tr, loss_te



def cross_validation_demo_ridge(y, tx, k_fold, lambdas, degrees):
    """cross validation over regularisation parameter lambda and hyperparameter degree.
    
    Args:
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
        degrees: list of degrees for polynomial expansion
    Returns:
        best_degree : integer, value of the best degree
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best pair (lambda,degree)
    """
    
    seed = 12
    k_fold = k_fold
    lambdas = lambdas
    # Split data in k fold
    k_idx = build_k_indices(y, k_fold, seed)
    
    # Define matrices to store loss of test data

    rmse_tr = np.zeros((len(lambdas),len(degrees)))
    rmse_te = np.zeros((len(lambdas),len(degrees)))
    
    for i,param in enumerate(lambdas):
        for j,deg in enumerate(degrees):
            l_tr = 0
            l_te = 0
            for k in range(k_fold):
                loss_tr, loss_te = cross_validation_ridge(y,tx, k_idx, k,param, deg)
                l_te += loss_te
                l_tr += loss_tr
            l_te = l_te/k_fold
            l_tr = l_tr/k_fold
            rmse_tr[i,j] = l_tr
            rmse_te[i,j] = l_te
    
    idx_lambda,idx_degree = np.unravel_index(np.argmin(rmse_te),rmse_te.shape)
    best_degree,best_lambda,best_rmse = degrees[idx_degree],lambdas[idx_lambda],rmse_te[idx_lambda,idx_degree]
    
    if len(degrees) == 1:
        cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    
    print("The choice of lambda which leads to the best test rmse is %.5f with a test rmse of %.3f. The best degree is %.1f" % (best_lambda, best_rmse,best_degree))
    return best_degree, best_lambda, best_rmse

