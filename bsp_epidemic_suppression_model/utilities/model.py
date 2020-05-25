"""
Data from the literature used in calculations
"""
from bsp_epidemic_suppression_model.utilities.distributions import (
    lognormal_cdf,
    gamma_density,
)

# R0, base value and parameters
R0 = 1
r0_alpha = 4.865916955  # TODO: add source
r0_beta = 0.6487889273


def r0(tau: float) -> float:
    """Effective default reproduction number density."""
    return R0 * gamma_density(tau, r0_alpha, r0_beta)


# DATA FOR A TWO-COMPONENTS MODEL (asymptomatics and symptomatics)

# Proportion of individuals who are symptomatics. Source: https://science.sciencemag.org/highwire/markup/744126/expansion?width=1000&height=500&iframe=true&postprocessors=highwire_figures%2Chighwire_math%2Chighwire_embed
fraction_symptomatics = 0.6

# Contributions to R0. Source: same paper
contribution_of_symptomatics_to_R0 = 0.95

R0sy = (
    contribution_of_symptomatics_to_R0 / fraction_symptomatics * R0
    if fraction_symptomatics > 0
    else 0
)
R0asy = (
    (1 - contribution_of_symptomatics_to_R0) / (1 - fraction_symptomatics) * R0
    if fraction_symptomatics < 1
    else 0
)


def r0sy(tau: float):
    """Effective default reproduction number density for symptomatic individuals."""
    return R0sy * gamma_density(tau, r0_alpha, r0_beta)


def r0asy(tau: float):
    """Effective default reproduction number density for asymptomatic individuals."""
    return R0asy * gamma_density(tau, r0_alpha, r0_beta)


def make_scenario_parameters_for_asymptomatics_symptomatics_model():
    """Returns the lists p_gs and r0_gs for the two-components model for the severity, namely for asymptomatic and
    symptomatic individuals."""
    p_gs = [1 - fraction_symptomatics, fraction_symptomatics]
    r0_gs = [lambda t, tau: r0asy(tau), lambda t, tau: r0sy(tau)]

    return p_gs, r0_gs


# Incubation period distribution, and parameters
incubation_mu = 1.644
incubation_sigma = 0.363


def FS(tau: float, normalization: float = 1.0) -> float:
    """
    Symptoms onset distribution. Possibly improper (if normalization < 1).
    """
    if tau > 0:
        # return 1
        return normalization * lognormal_cdf(tau, incubation_mu, incubation_sigma)
    return 0


def effectiveness_from_R(R: float):
    """
    Trivial effectiveness expressed as fraction of R reduction, in [0, 1] from min to max effectiveness
    """
    return 1.0 - R / R0
