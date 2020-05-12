import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.stats import gamma, lognorm, norm
import scipy.integrate as integrate
from time import sleep

def plot_functions(fs, x_min, x_max, step):
    x_values = np.arange(x_min, x_max, step)
    for i, f in enumerate(fs):
        plt.plot(x_values, [f(x) for x in x_values], label=str(i))
    plt.legend()
    plt.show()