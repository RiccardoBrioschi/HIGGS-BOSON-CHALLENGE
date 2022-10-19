import numpy as np
from implementations import *
from costs import *
from plots import cross_validation_visualization
from preprocessing import *

def build_k_indices_log(y, k_fold, seed):
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



def cross_validation_demo_log(y, x, k_fold, lambdas, gamma, initial_w, max_iters):
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
    k_indices = build_k_indices_log(y, k_fold, seed)
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



def build_k_indices_r(y, k_fold, seed):
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


def cross_validation_r(y, x, k_indices, k, lambda_, degree):
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

    x_train_temp = x_train[:,1:-4]
    x_test_temp = x_test[:, 1:-4]

    phi = build_poly(x_train_temp, degree)
    phi2 = build_poly(x_test_temp, degree)
    
    #aggiungo offset

    _, phi = build_model_data(y_train, phi)
    _, phi2 = build_model_data(y_test, phi2)

    #aggiungo categoriche
    phi=np.hstack((phi,x_train[:,:-4]))
    phi2=np.hstack((phi2,x_test[:,:-4]))

    w_opt, _ = ridge_regression(y_train, phi, lambda_)

    mse_test = compute_loss_linear_regression(y_train,phi, w_opt)
    mse_train = compute_loss_linear_regression(y_test,phi2, w_opt)
    loss_tr = np.sqrt(2*mse_train)
    loss_te = np.sqrt(2*mse_test)

    return loss_tr, loss_te



def cross_validation_demo_r(y, tx, k_fold, lambdas, degree):
    """cross validation over regularisation parameter lambda.
    
    Args:
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
    k_indices = build_k_indices_r(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # ***************************************************
    for j in lambdas:
        l_tr = 0
        l_te = 0       
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation_r(y, tx, k_indices, k, j, degree)
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
    
    print("The choice of lambda which leads to the best test rmse is %.5f with a test rmse of %.3f" % (best_lambda, best_rmse))
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
    k_indices = build_k_indices_r(y, k_fold, seed)
    
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
                loss_tr, loss_te = cross_validation_r(y, tx, k_indices, k, j, d)
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
