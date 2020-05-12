import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.stats import gamma, lognorm, norm
import scipy.integrate as integrate
from time import sleep


@dataclass
class DeltaMeasure:
    position: float
    height: float


def convolve(f1, f2, x_min, x_max, step):
    if isinstance(f2, DeltaMeasure):
        return lambda x: f2.height * f1(x - f2.position)
    raise ValueError("Not implemented yet")  # TODO: add

def list_from_f(f, x_min, x_max, step):
    x_values = np.arange(x_min, x_max + step, step)
    return [f(x) for x in x_values]


def f_from_list(f_values, x_min, x_max, step):
    def f(x):
        if x < x_min:
            return f_values[0]
        if x > x_max:
            return f_values[-1]
        i = int((x - x_min) / step)
        return f_values[i]

    return f