from matplotlib import pyplot as plt
import math

from bsp_epidemic_suppression_model.algorithm.model_blocks import effectiveness_from_R
from bsp_epidemic_suppression_model.model_utilities.epidemic_data import (
    make_scenario_parameters_for_asymptomatic_symptomatic_model,
    rho0,
    R0,
)
from bsp_epidemic_suppression_model.model_utilities.scenario import (
    make_homogeneous_scenario,
)
from bsp_epidemic_suppression_model.math_utilities.functions_utils import (
    DeltaMeasure,
    RealRange,
    integrate,
)

from bsp_epidemic_suppression_model.algorithm.time_evolution_main_function import (
    compute_time_evolution,
)

import warnings

warnings.filterwarnings("ignore")


tau_max = 30
integration_step = 0.1


def dependency_on_testing_timeliness_homogeneous_model_example():
    """
    Example of several computation of the limit R_t in homogeneous scenarios (i.e. with no app usage) in which the time
    interval Δ^{A → T} varies from 0 to 10 days.
    """
    n_iterations = 8

    # gs = [asymptomatic, symptomatic]
    p_gs, beta0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()

    DeltaAT_values_list = list(range(0, 10))
    Effinfty_values_list = []

    for DeltaAT in DeltaAT_values_list:

        scenario = make_homogeneous_scenario(
            p_gs=p_gs,
            beta0_gs=beta0_gs,
            t_0=0,
            ss=[0, 0.5],
            sc=0.7,
            xi=0.9,
            p_DeltaAT=DeltaMeasure(position=DeltaAT),
        )

        step_data_list = compute_time_evolution(
            scenario=scenario,
            real_range=RealRange(0, tau_max, integration_step),
            n_iterations=n_iterations,
            verbose=False,
        )

        Rinfty = step_data_list[-1].R
        Effinfty = effectiveness_from_R(Rinfty)
        Effinfty_values_list.append(Effinfty)

    fig = plt.figure(figsize=(10, 15))

    Rinfty_plot = fig.add_subplot(111)
    Rinfty_plot.set_xlabel("Δ^{A → T} (days)")
    Rinfty_plot.set_ylabel("Eff_infty")
    Rinfty_plot.grid(True)
    Rinfty_plot.set_xlim(0, DeltaAT_values_list[-1])
    Rinfty_plot.set_ylim(0, 0.6)
    Rinfty_plot.plot(DeltaAT_values_list, Effinfty_values_list, color="black",),

    plt.show()


def dependency_on_share_of_symptomatics_homogeneous_model_example():

    p_sym_list = [0.3, 0.4, 0.5, 0.6, 0.7]
    x_axis_list = []
    Effinfty_values_list = []

    for p_sym in p_sym_list:

        contribution_of_symptomatics_to_R0 = 1 - math.exp(
            -4.993 * p_sym
        )  # Random choice, gives 0.95 when p_sym = 0.6
        R0_sym = contribution_of_symptomatics_to_R0 / p_sym * R0
        R0_asy = (1 - contribution_of_symptomatics_to_R0) / (1 - p_sym) * R0

        # gs = [asymptomatic, symptomatic]
        p_gs, beta0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model(
            p_sym=p_sym,
            contribution_of_symptomatics_to_R0=contribution_of_symptomatics_to_R0,
        )
        n_iterations = 6

        scenario = make_homogeneous_scenario(
            p_gs=p_gs,
            beta0_gs=beta0_gs,
            t_0=0,
            ss=[0, 0.5],
            sc=0.7,
            xi=0.9,
            p_DeltaAT=DeltaMeasure(position=2),
        )

        step_data_list = compute_time_evolution(
            scenario=scenario,
            real_range=RealRange(0, tau_max, integration_step),
            n_iterations=n_iterations,
            verbose=False,
        )

        Rinfty = step_data_list[-1].R
        Effinfty = effectiveness_from_R(Rinfty)

        x_axis_list.append(
            f"p_sym = {p_sym},\nR^0_sym={round(R0_sym, 2)},\nR^0_asy={round(R0_asy, 2)}"
        )
        Effinfty_values_list.append(Effinfty)

    plt.xticks(p_sym_list, x_axis_list, rotation=0)
    plt.xlim(0, p_sym_list[-1])
    plt.ylim(0, 0.8)
    plt.grid(True)
    plt.plot(p_sym_list, Effinfty_values_list, color="black",),

    plt.show()


def dependency_on_infectiousness_width_homogeneous_model_example():

    infectiousness_rescale_factors = [0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8]

    expected_default_generation_times_list = []
    Effinfty_values_list = []

    for f in infectiousness_rescale_factors:

        def rescaled_rho0(tau):
            return (1 / f) * rho0(tau / f)

        assert round(integrate(rescaled_rho0, 0, tau_max), 5) == 1

        EtauC0 = integrate(
            lambda tau: tau * rescaled_rho0(tau), 0, tau_max
        )  # Expected default generation time
        expected_default_generation_times_list.append(EtauC0)

        # gs = [asymptomatic, symptomatic]
        p_gs, beta0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model(
            rho0=rescaled_rho0
        )
        n_iterations = 6

        scenario = make_homogeneous_scenario(
            p_gs=p_gs,
            beta0_gs=beta0_gs,
            t_0=0,
            ss=[0, 0.5],
            sc=0.7,
            xi=0.9,
            p_DeltaAT=DeltaMeasure(position=2),
        )

        step_data_list = compute_time_evolution(
            scenario=scenario,
            real_range=RealRange(0, tau_max, integration_step),
            n_iterations=n_iterations,
            verbose=False,
        )

        Rinfty = step_data_list[-1].R
        Effinfty = effectiveness_from_R(Rinfty)

        Effinfty_values_list.append(Effinfty)

    plt.ylim(0, 0.8)
    plt.grid(True)
    plt.plot(
        expected_default_generation_times_list, Effinfty_values_list, color="black",
    ),
    plt.xlabel("E(τ^C)")
    plt.ylabel("Eff_∞")
    plt.title("Effectiveness under rescaling of the generation time τ^C distribution")

    plt.show()


if __name__ == "__main__":
    # dependency_on_testing_timeliness_homogeneous_model_example()
    # dependency_on_share_of_symptomatics_homogeneous_model_example()
    dependency_on_infectiousness_width_homogeneous_model_example()
