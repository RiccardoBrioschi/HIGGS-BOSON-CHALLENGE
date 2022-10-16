
import matplotlib.pyplot as plt
import numpy as np
from costs import *
from helpers import *
from gradients import *


def choose_max_iter(y,tx,initial_w,max_iters, gamma, method = 'MSE'):

    """
    Function to choose the best number of iterations for the gradient descent.
    """
    losses = np.zeros(max_iters)
    w = initial_w

    for n in range(max_iters):
        if method == 'MSE':
            grad = compute_gradient_linear_regression(y,tx,w)
        if method == 'logistic_regression':
            grad = compute_gradient_logistic_regression(y,tx,w)
        w = w - gamma*grad
        if method == 'MSE':
            loss = compute_loss_linear_regression(y,tx,w)
        if method == 'logistic_regression':
            loss = compute_logloss_logistic_regression(y,tx,w)
        losses[n]=loss

    print(np.min(losses))
    plt.plot(list(range(max_iters)),losses, '-ro',linewidth = 0.2, markersize = 0.2)
    plt.title('Andamento della loss')

def cross_validation_visualization(lambds, rmse_tr, rmse_te):
    """visualization the curves of rmse_tr and rmse_te."""
    plt.semilogx(lambds, rmse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, rmse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("r mse")
    #plt.xlim(1e-4, 1)
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")