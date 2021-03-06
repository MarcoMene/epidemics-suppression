"""
Here we collect some PDFs and CDFs used in the computations.
"""
import numpy as np
from scipy.stats import gamma, norm


def gamma_pdf(x: float, alpha: float, beta: float) -> float:
    """
    PDF of gamma distribution.
    """
    return gamma.pdf(x, a=alpha, scale=1 / beta)


def gamma_cdf(x: float, alpha: float, beta: float) -> float:
    """
    CDF of gamma distribution.
    """
    return gamma.cdf(x, a=alpha, scale=1 / beta)


def lognormal_cdf(x: float, mu: float, sigma: float) -> float:
    """
    CDF of log-normal distribution.
    """
    return norm.cdf((np.log(x) - mu) / sigma)


def weibull_pdf(x: float, k: float, lambda_: float) -> float:
    """
    PDF of Weibull distribution with shape k and scale lambda_.
    """
    return k / lambda_ * (x / lambda_) ** (k - 1) * np.exp(-((x / lambda_) ** k))
