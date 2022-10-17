import numpy as np
from implementations import reg_logistic_regression
from costs import *
from plots import cross_validation_visualization

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
    """

    train_indices = np.concatenate([k_indices[:k].ravel(),k_indices[k+1:].ravel()])
    test_indices = k_indices[k]
    x_test = x[test_indices]
    y_test = y[test_indices]
    x_train = x[train_indices]
    y_train = y[train_indices]

    w_opt,_ = reg_logistic_regression(y_train,x_train,lambda_,initial_w,max_iters,gamma)

    loss_tr = compute_logloss_logistic_regression(y_train,x_train,w_opt)
    loss_te = compute_logloss_logistic_regression(y_test,x_test,w_opt)

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
    
    seed = 10
    w = initial_w
    k_fold = k_fold
    lambdas = lambdas
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    loss_tr = []
    loss_te = []
    for lambda_ in lambdas:
        temp_te, temp_tr = 0,0
        for k in range(k_fold):
            errors = cross_validation_log(y, x, k_indices, k, lambda_, gamma, w, max_iters)
            temp_tr+= errors[0]
            temp_te+= errors[1]
        loss_tr.append(temp_tr/k_fold)
        loss_te.append(temp_te/k_fold)
    best_loss = min(loss_te)
    best_lambda = lambdas[loss_te.index(best_loss)]

    cross_validation_visualization(lambdas, loss_tr, loss_te)
    return best_lambda, best_loss


