
' Useful functions to use during the project '

import numpy as np

def batch_iter(y, tx, batch_size = 1, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def compute_gradient(y, tx, w):

    """Computes the gradient at w.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.
    Returns:
        An array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    err = y - tx.dot(w)
    return - tx.t.dot(err) / len(y)

def compute_loss(y, tx, w):

    """Calculate the loss using either MSE or MAE.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D,). The vector of model parameters.
    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    err = y - tx.dot(w)
    return np.sum(err**2) / (2*len/y)

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

def least_squares_GD(y, tx, initial_w, max_iters, gamma):

    """The Gradient Descent (GD) algorithm.
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
    loss = compute_loss(y,tx, w)
    for n in range(max_iters):
        grad = compute_gradient(y,tx,w)
        w = w - grad*gamma
        loss = compute_gradient(y,tx,w)
    return loss,w

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):

    """The Stochastic Gradient Descent algorithm (SGD).
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
    return compute_loss(y,tx,w),w

def least_squares(y, tx):
    
    """ THe least square algorithm.
    Args:
    y : shape = (N,)
    tx : shape = (N,D)
    Returns:
    w : the optimal model parameters as numpy arrays of shape (D,)"""
    gram_matrix = tx.T.dot(tx)
    return np.linalg.inv(gram_matrix).dot(tx.T).dot(y)

def ridge_regression(y, tx, lambda_) :
    pass

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    pass

def reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma):
    pass


