from bsp_epidemic_suppression_model.utilities.epidemic_data import beta0, FS
from bsp_epidemic_suppression_model.algorithm.model_blocks import (
    suppressed_r_from_test_cdf,
)
from bsp_epidemic_suppression_model.utilities.plotting_utils import plot_functions
from bsp_epidemic_suppression_model.utilities.functions_utils import RealRange
import scipy.integrate as integrate
from numpy import heaviside

tau_max = 30
step = 0.05


def r0_suppression_with_fixed_testing_time():
    tau_s = 10

    FT = lambda tau: heaviside(tau - tau_s, 1)
    xi = 1.0  # Probability of (immediate) isolation given positive test

    suppressed_r_0 = suppressed_r_from_test_cdf(beta0, FT, xi)
    suppressed_R_0 = integrate.quad(lambda tau: suppressed_r_0(tau), 0, tau_max)[0]

    print("suppressed R_0 =", suppressed_R_0)
    plot_functions(
        [beta0, suppressed_r_0], RealRange(x_min=0, x_max=tau_max, step=step)
    )


def r0_suppression_due_to_symptoms_only():
    Deltat_test = 4
    ss = 0.2

    FT = lambda tau: ss * FS(tau - Deltat_test)
    xi = 1.0  # Probability of (immediate) isolation given positive test

    suppressed_r_0 = suppressed_r_from_test_cdf(beta0, FT, xi)
    suppressed_R_0 = integrate.quad(lambda tau: suppressed_r_0(tau), 0, tau_max)[0]

    print("suppressed R_0 =", suppressed_R_0)
    plot_functions(
        [beta0, suppressed_r_0], RealRange(x_min=0, x_max=tau_max, step=step)
    )


if __name__ == "__main__":

    r0_suppression_due_to_symptoms_only()
