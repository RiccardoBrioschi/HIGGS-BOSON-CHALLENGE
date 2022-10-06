
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_hist(tx,labels):
    for idx,name in enumerate(labels):
        plt.figure()
        plt.hist(tx[:,idx], label = name, alpha = 0.3, density = True, bins = 30 )
        plt.legend()
        plt.xlabel(name)
        plt.ylabel('Density')