"""
This file contains epidemic data specific of COVID-19 used in the calculations as inputs.
Note that we normalize the "default" effective reproduction number R0 to 1, as we are only interested in the relative
suppression. On the other hand, the breakdown of R0 in components and the shape of the infectiousness beta0 are taken
from the literature, as well as the incubation period distribution.
Below we consider a "two components model", in which the infected individuals are only divided, by illness severity,
into symptomatic and asymptomatic.
The source for all the numeric values is
https://science.sciencemag.org/content/368/6491/eabb6936
"""
from typing import Tuple, List, Callable

from bsp_epidemic_suppression_model.math_utilities.distributions import (
    lognormal_cdf,
    weibull_pdf,
)


# Default effective reproduction number
R0 = 1
k, lambda_ = (
    2.855,
    5.611,
)  # These parameters give a Weibull distribution with mean 5.00 and variance 3.61


def rho0(tau: float) -> float:
    """Default generation time distribution."""
    return weibull_pdf(tau, k, lambda_)


def beta0(tau: float) -> float:
    """Default infectiousness."""
    return R0 * rho0(tau)


# Incubation period distribution

incubation_mu = 1.644
incubation_sigma = 0.363


def FS(tau: float) -> float:
    """
    Cumulative distribution of the time of symptoms onset.
    """
    if tau > 0:
        return lognormal_cdf(tau, incubation_mu, incubation_sigma)
    return 0


# Data for the "two-components model" (asymptomatic and symptomatic individuals)

p_sym = 0.6  # Fraction of infected individuals who are symptomatic.
contribution_of_symptomatics_to_R0 = (
    0.95  # Contributions of symptomatic individuals to R0.
)


def make_scenario_parameters_for_asymptomatic_symptomatic_model(
    rho0: Callable[[float], float] = rho0,
    p_sym: float = p_sym,
    contribution_of_symptomatics_to_R0: float = contribution_of_symptomatics_to_R0,
) -> Tuple[List[float], List[Callable[[float, float], float]]]:
    """
    Returns the lists p_gs and β0_gs for the "two-components model" for the severity, namely for asymptomatic and
    symptomatic individuals.
    """
    p_asy = 1 - p_sym  # Fraction of infected individuals who are asymptomatic.

    R0_asy = (  # Component of R0 due to asymptomatic individuals
        (1 - contribution_of_symptomatics_to_R0) / p_asy * R0 if p_asy > 0 else 0
    )
    R0_sym = (  # Component of R0 due to symptomatic individuals
        contribution_of_symptomatics_to_R0 / p_sym * R0 if p_sym > 0 else 0
    )

    assert R0 == p_sym * R0_sym + p_asy * R0_asy

    def beta0_asy(tau: float):
        """Default infectiousness for asymptomatic individuals."""
        return R0_asy * rho0(tau)

    def beta0_sym(tau: float):
        """Default infectiousness for symptomatic individuals."""
        return R0_sym * rho0(tau)

    p_gs = [1 - p_sym, p_sym]
    beta0_gs = [lambda t, tau: beta0_asy(tau), lambda t, tau: beta0_sym(tau)]

    return p_gs, beta0_gs



