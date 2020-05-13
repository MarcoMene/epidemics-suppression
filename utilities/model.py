# Corrected reproduction number


from utilities.distributions import lognormal_cdf, gamma_density


alpha = 0.6  # Proportion of individuals who are symptomatics. Source: https://science.sciencemag.org/highwire/markup/744126/expansion?width=1000&height=500&iframe=true&postprocessors=highwire_figures%2Chighwire_math%2Chighwire_embed




# R0, base value and parameters
R0 = 1
r0_alpha = 4.865916955
r0_beta = 0.6487889273


def r0(tau: float) -> float:
    """
    Effective reproduction number density.
    """
    return R0 * gamma_density(tau, r0_alpha, r0_beta)

sy_contr_to_R = 0.95

R0sy = sy_contr_to_R / alpha * R0
R0asy = (1 - sy_contr_to_R) / (1 - alpha) * R0

def r0sy(tau: float):
    return R0sy * gamma_density(tau, r0_alpha, r0_beta)

def r0asy(tau: float):
    return R0asy * gamma_density(tau, r0_alpha, r0_beta)


# Incubation period distribution, and parameters
incubation_mu = 1.644
incubation_sigma = 0.363


def FS(tau: float, normalization: float = 1.) -> float:
    """
    Symtoms onset distribution. Possibly improper (if normalization < 1).
    """
    if tau > 0:
        return 1
        # return normalization * lognormal_cdf(tau, incubation_mu, incubation_sigma)
    return 0




# def FAs(tau: float, sS: float=1.) -> float:
#     return sS * FS(tau)


def suppressed_r_from_test_cdf(r: callable, F_T: callable, xi: float) -> callable:
    """
    Given a starting r0 density and a test CDF, calculates the new r profile, suppressed
    """
    return lambda tau: r(tau) * (1 - F_T(tau) * xi)


def effectiveness_from_R(R: float):
    """
    Trivial effectiveness expressed as fraction of R reduction, in [0, 1] from min to max effectiveness
    """
    return 1. - R / R0
