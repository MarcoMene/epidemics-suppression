import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.stats import gamma, lognorm, norm
import scipy.integrate as integrate
from time import sleep

def gamma_density(x, alpha, beta):
    return gamma.pdf(x, a=alpha, scale=1/beta)

def lognormal_cdf(x, mu, sigma):
    return norm.cdf((np.log(x) - mu)/sigma)

