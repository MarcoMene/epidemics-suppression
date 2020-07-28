import scipy.integrate as integrate
from numpy import heaviside

from bsp_epidemic_suppression_model.model_utilities.epidemic_data import beta0, FS

from bsp_epidemic_suppression_model.math_utilities.plotting_utils import plot_functions
from bsp_epidemic_suppression_model.math_utilities.functions_utils import RealRange

from bsp_epidemic_suppression_model.algorithm.model_blocks import (
    suppressed_beta_from_test_cdf,
)


tau_max = 30
step = 0.05


def R_suppression_with_fixed_testing_time():
    """
    Example computing the suppressed beta and R, given that the testing time CDF F^T is a step function.
    """
    tau_s = 10

    FT = lambda tau: heaviside(tau - tau_s, 1)  # CDF of testing time
    xi = 1.0  # Probability of (immediate) isolation given positive test

    suppressed_beta_0 = suppressed_beta_from_test_cdf(beta0, FT, xi)
    suppressed_R_0 = integrate.quad(lambda tau: suppressed_beta_0(tau), 0, tau_max)[0]

    print("suppressed R_0 =", suppressed_R_0)
    plot_functions(
        [beta0, suppressed_beta_0], RealRange(x_min=0, x_max=tau_max, step=step)
    )


def R_suppression_due_to_symptoms_only():
    """
    Example computing the suppressed beta and R, given that the testing time CDF F^T is just a traslation and rescaling
    of the symptoms onset CDF F^S.
    """
    Deltat_test = 4
    ss = 0.2

    FT = lambda tau: ss * FS(tau - Deltat_test)  # CDF of testing time
    xi = 1.0  # Probability of (immediate) isolation given positive test

    suppressed_beta_0 = suppressed_beta_from_test_cdf(beta0, FT, xi)
    suppressed_R_0 = integrate.quad(lambda tau: suppressed_beta_0(tau), 0, tau_max)[0]

    print("suppressed R_0 =", suppressed_R_0)
    plot_functions(
        [beta0, suppressed_beta_0], RealRange(x_min=0, x_max=tau_max, step=step)
    )
