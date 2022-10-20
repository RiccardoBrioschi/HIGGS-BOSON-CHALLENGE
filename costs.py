
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
    N = len(y)
    loss = np.sum(-tx.dot(w)*y + np.log(1+np.exp(tx.dot(w))))
    return loss / N

def compute_mse(y, tx, w):
    """compute the loss by mse.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: weights, numpy array of shape(D,), D is the number of features.
    
    Returns:
        mse: scalar corresponding to the mse with factor (1 / 2 n) in front of the sum

    """
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse
