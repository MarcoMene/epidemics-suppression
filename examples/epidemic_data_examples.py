"""
Some examples plotting the distributions of epidemic data used as inputs of the model.
"""

from scipy import integrate as integrate

from bsp_epidemic_suppression_model.math_utilities.functions_utils import (
    RealRange,
    integrate,
)
from bsp_epidemic_suppression_model.math_utilities.plotting_utils import plot_functions
from bsp_epidemic_suppression_model.model_utilities.epidemic_data import FS, beta0


def plot_symptoms_onset_distribution():
    tau_max = 30
    step = 0.05

    plot_functions(
        [FS],
        RealRange(x_min=0, x_max=tau_max, step=step),
        labels=["FS"],
        title="The CDF of tau^S",
    )
    EtauS = integrate(
        lambda tau: (1 - FS(tau)), 0, tau_max
    )  # Expected time of symptoms onset for symptomatics
    print("E(tauS) =", EtauS)


def plot_and_integrate_beta0():
    tau_max = 30
    step = 0.05

    print("The function beta^0_0:")
    plot_functions([beta0], RealRange(x_min=0, x_max=tau_max, step=step))

    print(
        "Integral of beta^0_0:", integrate.quad(beta0, 0, 30)[0]
    )  # Should (approximately) give back R0
