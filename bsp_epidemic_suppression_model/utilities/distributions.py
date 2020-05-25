import numpy as np
from scipy.stats import gamma, norm

"""
Wrappers of scipy distributions.
"""


def gamma_density(x: float, alpha: float, beta: float) -> float:
    """"
    PDF of Gamma distribution.
    """
    return gamma.pdf(x, a=alpha, scale=1 / beta)


def gamma_cdf(x: float, alpha: float, beta: float) -> float:
    """"
    CDF of Gamma distribution.
    """
    return gamma.cdf(x, a=alpha, scale=1 / beta)


def lognormal_cdf(x: float, mu: float, sigma: float) -> float:
    """"
    CDF of Log-Normal distribution.
    """
    return norm.cdf((np.log(x) - mu) / sigma)
