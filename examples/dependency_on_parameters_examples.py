from matplotlib import pyplot as plt
import numpy as np
import math

from bsp_epidemic_suppression_model.algorithm.model_blocks import effectiveness_from_R
from bsp_epidemic_suppression_model.model_utilities.epidemic_data import (
    make_scenario_parameters_for_asymptomatic_symptomatic_model,
    rho0,
    R0,
)
from bsp_epidemic_suppression_model.model_utilities.scenario import (
    Scenario,
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
    Example of several computations of the limit Eff_∞ in homogeneous scenarios (i.e. with no app usage)
    in which the time interval Δ^{A → T} varies from 0 to 10 days.
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
    Rinfty_plot.set_ylabel("Eff_∞")
    Rinfty_plot.grid(True)
    Rinfty_plot.set_xlim(0, DeltaAT_values_list[-1])
    Rinfty_plot.set_ylim(0, 0.6)
    Rinfty_plot.plot(DeltaAT_values_list, Effinfty_values_list, color="black",),

    plt.show()


def dependency_on_share_of_symptomatics_homogeneous_model_example():
    """
    Example of several computations of the limit Eff_∞ in homogeneous scenarios (i.e. with no app usage)
    in which the fraction p_sym of symptomatic individuals and their contribution to R^0 vary.
    """

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
            f"p_sym = {p_sym},\nκ={round(contribution_of_symptomatics_to_R0, 2)},\nR^0_sym={round(R0_sym, 2)},\nR^0_asy={round(R0_asy, 2)}"
        )
        Effinfty_values_list.append(Effinfty)

    plt.xticks(p_sym_list, x_axis_list, rotation=0)
    plt.xlim(0, p_sym_list[-1])
    plt.ylim(0, 0.8)
    plt.ylabel("Eff_∞")
    plt.grid(True)
    plt.plot(p_sym_list, Effinfty_values_list, color="black",),

    plt.show()


def dependency_on_infectiousness_width_homogeneous_model_example():
    """
    Example of several computations of the limit Eff_∞ in homogeneous scenarios (i.e. with no app usage)
    in which the default distribution ρ^0 of the generation time is rescaled by different factors.
    """

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
    plt.xlabel("E(τ^{0,C})")
    plt.ylabel("Eff_∞")
    plt.title(
        "Effectiveness under rescaling of the default generation time distribution"
    )

    plt.show()


def dependency_on_efficiencies_example():
    """
    Example of several computations of the limit Eff_∞ with app usage, where the parameters s^{s,app} and s^{c,app}
    vary.
    """
    n_iterations = 8

    # gs = [asymptomatic, symptomatic]
    p_gs, beta0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()

    ssapp_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    scapp_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    Effinfty_values_list = []

    for scapp in scapp_list:
        Effinfty_values_list_for_scapp = []
        for ssapp in ssapp_list:

            scenario = Scenario(
                p_gs=p_gs,
                beta0_gs=beta0_gs,
                t_0=0,
                ssapp=[0, ssapp],
                ssnoapp=[0, 0.2],
                scapp=scapp,
                scnoapp=0.2,
                xi=0.9,
                papp=lambda t: 0.6,
                p_DeltaATapp=DeltaMeasure(position=2),
                p_DeltaATnoapp=DeltaMeasure(position=4),
            )

            step_data_list = compute_time_evolution(
                scenario=scenario,
                real_range=RealRange(0, tau_max, integration_step),
                n_iterations=n_iterations,
                verbose=False,
            )

            Rinfty = step_data_list[-1].R
            Effinfty = effectiveness_from_R(Rinfty)

            print(
                f"s^{{s,app}} = {ssapp}, s^{{c,app}} = {scapp}, Eff_∞ = {round(Effinfty, 2)}"
            )

            Effinfty_values_list_for_scapp.append(Effinfty)

        Effinfty_values_list.append(Effinfty_values_list_for_scapp)

    ssapp_values_array, scapp_values_array = np.meshgrid(ssapp_list, scapp_list)

    fig = plt.figure(figsize=(10, 15))
    ax = fig.gca(projection="3d")

    ax.plot_surface(
        ssapp_values_array, scapp_values_array, np.array(Effinfty_values_list),
    )

    ax.set_xlabel("s^{s,app}")
    ax.set_ylabel("s^{c,app}")
    ax.set_zlabel("Eff_∞")

    plt.show()


def dependency_on_app_adoption_example():
    """
    Example of several computations of the limit Eff_∞ with app usage, where the fraction p^app of app adopters varies.
    """
    n_iterations = 8

    # gs = [asymptomatic, symptomatic]
    p_gs, beta0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()

    papp_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    Effinfty_values_list = []

    for papp in papp_list:
        scenario = Scenario(
            p_gs=p_gs,
            beta0_gs=beta0_gs,
            t_0=0,
            ssapp=[0, 0.5],
            ssnoapp=[0, 0.2],
            scapp=0.7,
            scnoapp=0.2,
            xi=0.9,
            papp=lambda t: papp,
            p_DeltaATapp=DeltaMeasure(position=2),
            p_DeltaATnoapp=DeltaMeasure(position=4),
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
    Rinfty_plot.set_xlabel("p^app")
    Rinfty_plot.set_ylabel("Eff_∞")
    Rinfty_plot.grid(True)
    Rinfty_plot.set_xlim(0, 1)
    Rinfty_plot.set_ylim(0, 1)
    Rinfty_plot.plot(papp_list, Effinfty_values_list, color="black",),

    plt.show()
