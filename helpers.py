# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np


def load_data(path, default_missing_value = -999.0):
    """Load data function. 
    Arguments:
    path : path to find the file

    Returns
    ids : numpy array containing ids of the observed data
    tx : feature matrix
    y : prediction converted according to the rule {'b': 1, 's': 0}
    """
    ids = np.genfromtxt(path, delimiter = ',',usecols = [0], dtype = int, skip_header = 1)
    tx = np.genfromtxt(path,skip_header = 1, delimiter = ',', usecols = list(range(2,32)))
    y = np.genfromtxt(path, skip_header = 1, delimiter = ',', usecols = 1, converters = {1: lambda x: 1 if x == b'b' else 0}, dtype = int)
    # We now convert missing data to np.nan
    tx[tx == default_missing_value] = np.nan
    return ids,tx,y

def standardize(data):
    """ 
    This function standardizes the feature matrix.
    Returns:
    std_data : standardize data
    mean : mean of data
    std : standard deviation of data.
    """
    mean = data.mean(axis = 0)
    std_data = data - data.mean(axis = 0)
    std = data.std(axis = 0)
    std_data = data / data.std(axis = 0)
    return std_data, mean, std


def batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
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

def build_model_data(y, X_without_offset):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), X_without_offset]
    return y, tx