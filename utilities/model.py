# Corrected reproduction number


from utilities.distributions import lognormal_cdf, gamma_density

# R0
R0_0 = 1

def r0_0(tau, alpha=4.865916955, beta=0.6487889273):
    return R0_0 * gamma_density(tau, alpha, beta)


# Incubation period distribution
def FS(tau, mu=1.644, sigma=0.363, normalization=1.):
    if tau > 0:
        return normalization * lognormal_cdf(tau, mu, sigma)
    return 0


def FAs(tau, sS=1.):
    return sS * FS(tau)


def make_r_t_from_test_cdf(r0_t, F_T, xi):
    return lambda tau: r0_t(tau) * (1 - F_T(tau) * xi)


def make_r_from_test_cdf(r0, F_T, xi):
    return lambda t, tau: r0(t, tau) * (1 - F_T(t, tau) * xi)
