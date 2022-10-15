
'Useful functions to use during the project '

from errno import EEXIST
import numpy as np
from helpers import *
from costs import *
from gradients import *

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
        for y_minibatch, tx_minibatch in batch_iter(y,tx,1):
            stoch_grad = compute_stoch_gradient(y_minibatch,tx_minibatch,w)
            w = w - gamma*stoch_grad
    loss = compute_loss_linear_regression(y,tx,w)
    return w,loss

def least_squares(y, tx):
    
    """ The least square algorithm or linear regression using normal equations.
    Args:
    y : shape = (N,)
    tx : shape = (N,D)
    Returns:
    w : the optimal model parameters as numpy arrays of shape (D,)
    loss: the loss value (scalar) for least squares"""
        
    A = np.dot(tx.T,tx)
    b = np.dot(tx.T,y)
    w = np.linalg.solve(A,b)
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
    for n in range(max_iters):
        grad = compute_gradient_logistic_regression(y,tx,w)
        w = w - gamma*grad
    loss = compute_logloss_logistic_regression(y,tx,w)
    return w, loss

def reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma):

    """The Gradient Descent (GD) algorithm for regularized logistic regression.
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
    for n in range(max_iters):
        grad = compute_gradient_logistic_regression(y,tx,w) + lambda_*w/ N
        w = w - gamma*grad
    loss = compute_logloss_logistic_regression(y,tx,w)
    return w, loss


