
""" functions used to compute loss """

import numpy as np
from helpers import *

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
    return np.sum(err**2) / (2*len(y))

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


def compute_mse(y, tx, w):
    """compute the loss by mse.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: weights, numpy array of shape(D,), D is the number of features.
    
    Returns:
        mse: scalar corresponding to the mse with factor (1 / 2 n) in front of the sum

    >>> compute_mse(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), np.array([0.03947092, 0.00319628]))
    0.006417022764962313
    """
    
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse