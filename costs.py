
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