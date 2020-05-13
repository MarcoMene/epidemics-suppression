# Corrected reproduction number


from utilities.distributions import lognormal_cdf, gamma_density

# R0, base value and parameters
R0 = 1
r0_alpha = 4.865916955
r0_beta = 0.6487889273


def r0(tau: float) -> float:
    """
    Effective reproduction number density.
    """
    return R0 * gamma_density(tau, r0_alpha, r0_beta)


# Incubation period distribution, and parameters
incubation_mu = 1.644
incubation_sigma = 0.363


def FS(tau: float, normalization: float = 1.) -> float:
    """
    Symtoms onset distribution. Possibly improper (if normalization < 1).
    """
    if tau > 0:
        return normalization * lognormal_cdf(tau, incubation_mu, incubation_sigma)
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
