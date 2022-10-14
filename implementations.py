
'Useful functions to use during the project '

import numpy as np
from helpers import *

def compute_gradient_linear_regression(y, tx, w):

    """Computes the gradient at w for linear regression.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.
    Returns:
        An array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    err = y - tx.dot(w)
    return - tx.T.dot(err) / len(y)

def compute_gradient_logistic_regression(y,tx,w):

    """Computes the gradient at w for logistic regression.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.
    Returns:
        An array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    N = len(y)
    err = sigmoid(tx.dot(w)) - y
    return tx.T.dot(err) / N
    
def compute_loss_linear_regression(y, tx, w):

    """Calculate the loss using either MSE.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D,). The vector of model parameters.
    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    err = y - tx.dot(w)
    #return np.sum(err**2) / (2*len(y))
    return np.linalg.norm(err)**2 / (2*len(y))

def compute_logloss_logistic_regression(y, tx, w):
    
    """Calculate the loss for logistic regression.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D,). The vector of model parameters.
    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    predict = sigmoid(tx.dot(w))
    term = -y*np.log(predict) - (1-y)*np.log(1 - predict)
    return np.mean(term)

def compute_stoch_gradient(y, tx, w):

    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.
       This implementation holds whenever the number of batches is equal to 1.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(2, ). The vector of model parameters.
    Returns:
        An array of shape (D, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    for minibatch_y,minibatch_tx in batch_iter(y, tx):
        err = minibatch_y - minibatch_tx.dot(w)
        stoch_grad = -minibatch_tx.T.dot(err) / len(minibatch_y)
    return stoch_grad

def mean_squared_error_GD(y, tx, initial_w, max_iters, gamma):

    """The Gradient Descent (GD) algorithm for linear regression.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    Returns:
        loss: the loss value (scalar) for the final iteration of the method
        w: the model parameters as numpy arrays of shape (D, )
        """
    w = initial_w
    loss = compute_loss_linear_regression(y,tx, w)
    for n in range(max_iters):
        grad = compute_gradient_linear_regression(y,tx,w)
        w = w - grad*gamma
        loss = compute_loss_linear_regression(y,tx,w)
    return w,loss

def mean_squared_error_SGD(y, tx, initial_w, max_iters, gamma):

    """The Stochastic Gradient Descent algorithm (SGD) for linear regression.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
    Returns:
        loss: the loss value (scalar) for the last iteration of SGD
        w: the model parameters as numpy arrays of shape (D, ), for the final iteration of SGD
    """
    w = initial_w
    for n in range(max_iters):
        stoch_grad = compute_stoch_gradient(y,tx,w)
        w = w - gamma*stoch_grad
    return w,compute_loss_linear_regression(y,tx,w)

def least_squares(y, tx):
    
    """ The least square algorithm or linear regression using normal equations.
    Args:
    y : shape = (N,)
    tx : shape = (N,D)
    Returns:
    w : the optimal model parameters as numpy arrays of shape (D,)
    loss: the loss value (scalar) for least squares"""
        
    gram_matrix = np.dot(tx.T,tx)
    b = np.dot(tx.T,y)
    w = np.linalg.solve(gram_matrix,b)
    return w, compute_loss_linear_regression(y,tx,w)

def ridge_regression(y, tx, lambda_) :
    """Implement ridge regression using normal equations.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        ridge_loss: the loss value (scalar) for ridge regression.
    """
    lambda_tilde =  2 * lambda_ * len(y)
    A = tx.T.dot(tx) + lambda_tilde*np.eye(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(A,b)
    ridge_loss = compute_loss_linear_regression(y,tx,w) 
    return w, ridge_loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    
    """The Gradient Descent (GD) algorithm for logistic regression.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    Returns:
        loss: the loss value (scalar) for the final iteration of the method
        w: the model parameters as numpy arrays of shape (D, )
        """
    w = initial_w
    #losses = [compute_logloss_logistic_regression(y,tx,w)]
    for n in range(max_iters):
        grad = compute_gradient_logistic_regression(y,tx,w)
        w = w - gamma*grad
        #losses.append(compute_logloss_logistic_regression(y,tx,w))
        loss = compute_logloss_logistic_regression(y,tx,w)
    return w, loss

def reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma):

    """The Gradient Descent (GD) algorithm for logistic regression.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    Returns:
        loss: the loss value (scalar) for the final iteration of the method
        w: the model parameters as numpy arrays of shape (D, )
        """
    w = initial_w
    N = len(y)
    #losses = [compute_logloss_logistic_regression(y,tx,w)]
    for n in range(max_iters):
        grad = compute_gradient_logistic_regression(y,tx,w) + lambda_*w/ N
        w = w - gamma*grad
        #losses.append(compute_logloss_logistic_regression(y,tx,w))
        loss = compute_logloss_logistic_regression(y,tx,w)
    return w, loss
    pass


