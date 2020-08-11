"""
Examples running the complete algorithm in certain scenarios, and plotting the results.
"""

from bsp_epidemic_suppression_model.model_utilities.epidemic_data import (
    make_scenario_parameters_for_asymptomatic_symptomatic_model,
)
from bsp_epidemic_suppression_model.model_utilities.scenario import Scenario
from bsp_epidemic_suppression_model.math_utilities.functions_utils import (
    DeltaMeasure,
    RealRange,
)
from bsp_epidemic_suppression_model.model_utilities.plot_results import (
    plot_time_evolution,
)
from bsp_epidemic_suppression_model.algorithm.time_evolution_main_function import (
    compute_time_evolution,
)

import warnings

warnings.filterwarnings("ignore")


tau_max = 30
integration_step = 0.1


def time_evolution_homogeneous_model_optimistic_scenario_example():
    """
    Example in which there is no app usage, and the sensitivities s^S and s^C are quite high
    """

    # gs = [asymptomatic, symptomatic]

    n_iterations = 8

    p_gs, beta0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()

    ss = [0, 0.5]
    sc = 0.7
    xi = 0.9
    DeltaAT = 2

    scenario = Scenario(
        p_gs=p_gs,
        beta0_gs=beta0_gs,
        t_0=0,
        ssapp=[0, 0],
        ssnoapp=ss,
        scapp=0,
        scnoapp=sc,
        xi=xi,
        papp=lambda tau: 0,
        p_DeltaATapp=DeltaMeasure(position=0),
        p_DeltaATnoapp=DeltaMeasure(position=DeltaAT),
    )

    step_data_list = compute_time_evolution(
        scenario=scenario,
        real_range=RealRange(0, tau_max, integration_step),
        n_iterations=n_iterations,
        verbose=True,
    )

    plot_time_evolution(step_data_list=step_data_list, plot_components=False)


def time_evolution_optimistic_scenario_example():
    """
    Example in which the sensitivities s^S and s^C are high for who uses the app, and the app is used by 60% of the
    population.
    """

    # gs = [asymptomatic, symptomatic]

    n_iterations = 8

    p_gs, beta0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()

    scenario = Scenario(
        p_gs=p_gs,
        beta0_gs=beta0_gs,
        t_0=0,
        ssapp=[0, 0.8],
        ssnoapp=[0, 0.2],
        scapp=0.8,
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
        verbose=True,
    )

    plot_time_evolution(step_data_list=step_data_list)


def time_evolution_pessimistic_scenario_example():
    """
    Example in which the sensitivities s^S and s^C are pretty low, and the app is used by 60% of the population.
    """

    # gs = [asymptomatic, symptomatic]

    n_iterations = 8

    p_gs, beta0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()

    scenario = Scenario(
        p_gs=p_gs,
        beta0_gs=beta0_gs,
        t_0=0,
        ssapp=[0, 0.2],
        ssnoapp=[0, 0.2],
        scapp=0.5,
        scnoapp=0.2,
        xi=0.7,
        papp=lambda t: 0.6,
        p_DeltaATapp=DeltaMeasure(position=2),
        p_DeltaATnoapp=DeltaMeasure(position=4),
    )

    step_data_list = compute_time_evolution(
        scenario=scenario,
        real_range=RealRange(0, tau_max, integration_step),
        n_iterations=n_iterations,
        verbose=True,
    )

    plot_time_evolution(step_data_list=step_data_list)


def time_evolution_gradual_app_adoption_example():
    """
    Example in which the share of people using the app linearly increases, until reaching 60% in 30 days.
    """

    # gs = [asymptomatic, symptomatic]

    n_iterations = 16

    p_gs, beta0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()

    def papp(t: float) -> float:
        papp_infty = 0.6
        t_saturation = 30
        if 0 <= t < t_saturation:
            return papp_infty * t / t_saturation
        elif t >= t_saturation:
            return papp_infty

    scenario = Scenario(
        p_gs=p_gs,
        beta0_gs=beta0_gs,
        t_0=0,
        ssapp=[0, 0.8],
        ssnoapp=[0, 0.2],
        scapp=0.8,
        scnoapp=0.2,
        xi=0.9,
        papp=papp,
        p_DeltaATapp=DeltaMeasure(position=2),
        p_DeltaATnoapp=DeltaMeasure(position=4),
    )

    step_data_list = compute_time_evolution(
        scenario=scenario,
        real_range=RealRange(0, tau_max, integration_step),
        n_iterations=n_iterations,
        verbose=True,
    )

    plot_time_evolution(step_data_list=step_data_list)


if __name__ == "__main__":
    # time_evolution_optimistic_scenario_example()
    # time_evolution_pessimistic_scenario_example()
    # time_evolution_homogeneous_model_optimistic_scenario_example()
    time_evolution_gradual_app_adoption_example()
