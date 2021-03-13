from bsp_epidemic_suppression_model.algorithm_refactored.plot_results import (
    plot_time_evolution,
)
from bsp_epidemic_suppression_model.algorithm_refactored.time_evolution_main_function import (
    compute_time_evolution_approximated_algorithm,
)
from bsp_epidemic_suppression_model.math_utilities.config import UNITS_IN_ONE_DAY
from bsp_epidemic_suppression_model.math_utilities.discrete_distributions_utils import (
    DiscreteDistributionOnNonNegatives,
)
from bsp_epidemic_suppression_model.model_utilities.epidemic_data import (
    make_scenario_parameters_for_asymptomatic_symptomatic_model,
    tauS,
)
from bsp_epidemic_suppression_model.model_utilities.scenarios import (
    HomogeneousScenario,
    ScenarioWithApp,
)

p_gs, b0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()


def time_evolution_homogeneous_model_optimistic_scenario_example():
    """
    Example in which there is no app usage, and the sensitivities s^S and s^C are quite high
    """
    n_iterations = 8

    DeltaAT_in_days = 2

    scenario = HomogeneousScenario(
        p_gs=p_gs,
        b0_gs=b0_gs,
        tauS=tauS,
        t_0=0,
        ss=(lambda t: 0, lambda t: 0.5),
        sc=lambda t: 0.7,
        xi=lambda t: 0.9,
        DeltaAT=DiscreteDistributionOnNonNegatives(
            pmf_values=[1], tau_min=DeltaAT_in_days * UNITS_IN_ONE_DAY
        ),
    )

    step_data_list = compute_time_evolution_approximated_algorithm(
        scenario=scenario, n_iterations=n_iterations, verbose=True,
    )

    plot_time_evolution(step_data_list=step_data_list, plot_components=False)


def time_evolution_optimistic_scenario_example():
    """
    Example in which the sensitivities s^S and s^C are high for who uses the app,
    and the app is used by 60% of the population.
    """

    n_iterations = 8

    DeltaATapp_in_days = 2
    DeltaATnoapp_in_days = 4

    scenario = ScenarioWithApp(
        p_gs=p_gs,
        b0_gs=b0_gs,
        t_0=0,
        ssapp=(0, 0.8),
        ssnoapp=(0, 0.2),
        scapp=0.8,
        scnoapp=0.2,
        xi=0.9,
        papp=lambda t: 0.6,
        DeltaATapp=DiscreteDistributionOnNonNegatives(
            pmf_values=[1], tau_min=DeltaATapp_in_days * UNITS_IN_ONE_DAY
        ),
        DeltaATnoapp=DiscreteDistributionOnNonNegatives(
            pmf_values=[1], tau_min=DeltaATnoapp_in_days * UNITS_IN_ONE_DAY
        ),
    )

    step_data_list = compute_time_evolution_approximated_algorithm(
        scenario=scenario, n_iterations=n_iterations, verbose=True,
    )

    plot_time_evolution(step_data_list=step_data_list)


def time_evolution_pessimistic_scenario_example():
    """
    Example in which the sensitivities s^S and s^C are pretty low,
    and the app is used by 60% of the population.
    """

    n_iterations = 8

    DeltaATapp_in_days = 2
    DeltaATnoapp_in_days = 4

    scenario = ScenarioWithApp(
        p_gs=p_gs,
        beta0_gs=b0_gs,
        t_0=0,
        ssapp=(0, 0.2),
        ssnoapp=(0, 0.2),
        scapp=0.5,
        scnoapp=0.2,
        xi=0.7,
        papp=lambda t: 0.6,
        DeltaATapp=DiscreteDistributionOnNonNegatives(
            pmf_values=[1], tau_min=DeltaATapp_in_days * UNITS_IN_ONE_DAY
        ),
        DeltaATnoapp=DiscreteDistributionOnNonNegatives(
            pmf_values=[1], tau_min=DeltaATnoapp_in_days * UNITS_IN_ONE_DAY
        ),
    )

    step_data_list = compute_time_evolution_approximated_algorithm(
        scenario=scenario, n_iterations=n_iterations, verbose=True,
    )

    plot_time_evolution(step_data_list=step_data_list)


def time_evolution_gradual_app_adoption_example():
    """
    Example in which the share of people using the app linearly increases,
    until reaching 60% in 30 days.
    """

    n_iterations = 16

    DeltaATapp_in_days = 2
    DeltaATnoapp_in_days = 4

    def papp(t: float) -> float:
        papp_infty = 0.6
        t_saturation = 30
        if 0 <= t < t_saturation:
            return papp_infty * t / t_saturation
        elif t >= t_saturation:
            return papp_infty

    scenario = ScenarioWithApp(
        p_gs=p_gs,
        beta0_gs=b0_gs,
        t_0=0,
        ssapp=(0, 0.8),
        ssnoapp=(0, 0.2),
        scapp=0.8,
        scnoapp=0.2,
        xi=0.9,
        papp=papp,
        DeltaATapp=DiscreteDistributionOnNonNegatives(
            pmf_values=[1], tau_min=DeltaATapp_in_days * UNITS_IN_ONE_DAY
        ),
        DeltaATnoapp=DiscreteDistributionOnNonNegatives(
            pmf_values=[1], tau_min=DeltaATnoapp_in_days * UNITS_IN_ONE_DAY
        ),
    )

    step_data_list = compute_time_evolution_approximated_algorithm(
        scenario=scenario, n_iterations=n_iterations, verbose=True,
    )

    plot_time_evolution(step_data_list=step_data_list)


if __name__ == "__main__":
    time_evolution_homogeneous_model_optimistic_scenario_example()

    # time_evolution_optimistic_scenario_example()

    # time_evolution_gradual_app_adoption_example()

    pass
