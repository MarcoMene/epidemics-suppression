"""
Some examples printing and plotting the epidemic data used as inputs of the model.
"""

from bsp_epidemic_suppression_model.math_utilities.functions_utils import (
    RealRange,
    integrate,
)
from bsp_epidemic_suppression_model.math_utilities.plotting_utils import plot_functions
from bsp_epidemic_suppression_model.model_utilities.epidemic_data import (
    R0,
    rho0,
    beta0,
    FS,
)


def plot_generation_time():
    tau_max = 30
    step = 0.05

    print(
        "Expected default generation time: E(τ^C) =",
        integrate(lambda tau: tau * rho0(tau), 0, tau_max),
    )

    plot_functions(
        fs=[rho0],
        real_range=RealRange(x_min=0, x_max=tau_max, step=step),
        title="The default generation time distribution ρ^0",
    )


def plot_and_integrate_infectiousness():
    tau_max = 30
    step = 0.05

    integral_beta0 = integrate(beta0, 0, tau_max)
    print("The integral of β^0_0 is", integral_beta0)
    # This should (approximately) give back R0:
    assert round(integral_beta0 - R0, 5) == 0

    plot_functions(
        fs=[beta0],
        real_range=RealRange(x_min=0, x_max=tau_max, step=step),
        title="The default infectiousness β^0_0",
    )


def plot_symptoms_onset_distribution():
    tau_max = 30
    step = 0.05

    EtauS = integrate(
        lambda tau: (1 - FS(tau)), 0, tau_max
    )  # Expected time of symptoms onset for symptomatics
    print("E(τ^S) =", EtauS)

    plot_functions(
        fs=[FS],
        real_range=RealRange(x_min=0, x_max=tau_max, step=step),
        title="The CDF F^S of the symptoms onset time τ^S",
    )
