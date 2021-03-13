"""
Some examples of computations of the suppressed effective reproduction number, given the distribution of the time of
positive testing for an infected individual.
"""

import scipy.integrate as integrate
from numpy import heaviside

from bsp_epidemic_suppression_model.algorithm.model_blocks.testing_time_and_b_t_suppression import (
    compute_suppressed_b_t,
)
from bsp_epidemic_suppression_model.examples.plotting_utils import (
    plot_discrete_distributions,
    plot_functions,
)
from bsp_epidemic_suppression_model.math_utilities.config import (
    TAU_MAX_IN_UNITS,
    TAU_UNIT_IN_DAYS,
    UNITS_IN_ONE_DAY,
)
from bsp_epidemic_suppression_model.math_utilities.discrete_distributions_utils import (
    DiscreteDistributionOnNonNegatives,
    generate_discrete_distribution_from_cdf_function,
)
from bsp_epidemic_suppression_model.math_utilities.general_utilities import RealRange
from bsp_epidemic_suppression_model.model_utilities.epidemic_data import b0
from bsp_epidemic_suppression_model.old_stuff.epidemic_data import FS, beta0
from dev.old_stuff.algorithm.model_blocks import suppressed_beta_from_test_cdf

tau_max = 30
step = 0.05


def R_suppression_with_fixed_testing_time():
    """
    Example computing the suppressed beta and R, given that all individuals are tested
    at a given instant tau_s.
    """
    tau_s_in_days = 7

    tauT = DiscreteDistributionOnNonNegatives(
        pmf_values=[1], tau_min=tau_s_in_days * UNITS_IN_ONE_DAY, improper=True,
    )

    xi = 1.0  # Probability of (immediate) isolation given positive test

    suppressed_b0 = compute_suppressed_b_t(b0_t_gs=(b0,), tauT_t_gs=(tauT,), xi_t=xi,)[
        0
    ]
    suppressed_R_0 = suppressed_b0.total_mass

    print("suppressed R_0 =", suppressed_R_0)
    plot_discrete_distributions(ds=[b0, suppressed_b0], custom_labels=["β^0", "β"])


def R_suppression_due_to_symptoms_only():
    """
    Example computing the suppressed beta and R, given that the testing time CDF F^T is just a traslation and rescaling
    of the symptoms onset CDF F^S.
    """
    Deltat_test = 4
    ss = 0.2

    FT = lambda tau: ss * FS(tau - Deltat_test)  # CDF of testing time
    xi = 1.0  # Probability of (immediate) isolation given positive test

    tauT = generate_discrete_distribution_from_cdf_function(
        cdf=FT, tau_min=0, tau_max=TAU_MAX_IN_UNITS,
    )

    suppressed_b0 = compute_suppressed_b_t(b0_t_gs=(b0,), tauT_t_gs=(tauT,), xi_t=xi,)[
        0
    ]
    suppressed_R_0 = suppressed_b0.total_mass

    print("suppressed R_0 =", suppressed_R_0)
    plot_discrete_distributions(ds=[b0, suppressed_b0], custom_labels=["β^0", "β"])
