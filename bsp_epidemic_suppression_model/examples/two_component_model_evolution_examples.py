from bsp_epidemic_suppression_model.algorithm.evolution_with_app_algorithm import (
    compute_time_evolution_two_component,
)
from bsp_epidemic_suppression_model.examples.plot_results import plot_time_evolution
from bsp_epidemic_suppression_model.math_utilities.config import UNITS_IN_ONE_DAY
from bsp_epidemic_suppression_model.math_utilities.discrete_distributions_utils import (
    DiscreteDistributionOnNonNegatives,
)
from bsp_epidemic_suppression_model.model_utilities.epidemic_data import (
    make_scenario_parameters_for_asymptomatic_symptomatic_model,
    tauS,
)
from bsp_epidemic_suppression_model.model_utilities.scenarios import ScenarioWithApp

p_gs, b0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()


def time_evolution_two_component_model_optimistic_scenario_example():
    """
    Example in which there is no app usage, and the sensitivities s^S and s^C are quite high
    """
    t_max_in_days = 20

    DeltaAT_in_days = 2

    t_0 = 0

    scenario = ScenarioWithApp(
        p_gs=p_gs,
        b0_gs=b0_gs,
        tauS=tauS,
        t_0=0,
        ssapp=(lambda t: 0, lambda t: 0.5 if t >= t_0 else 0),
        ssnoapp=(lambda t: 0, lambda t: 0.5 if t >= t_0 else 0),
        scapp=lambda t: 0.7 if t >= t_0 else 0,
        scnoapp=lambda t: 0.7 if t >= t_0 else 0,
        xi=lambda t: 0.9 if t >= t_0 else 0,
        DeltaATapp=DiscreteDistributionOnNonNegatives(
            pmf_values=[1], tau_min=DeltaAT_in_days * UNITS_IN_ONE_DAY
        ),
        DeltaATnoapp=DiscreteDistributionOnNonNegatives(
            pmf_values=[1], tau_min=DeltaAT_in_days * UNITS_IN_ONE_DAY
        ),
        papp=lambda t: 0.5,
    )

    (
        t_in_days_list,
        nu,
        nu0,
        R,
        R_by_severity_app,
        R_by_severity_noapp,
        FT_infty,
    ) = compute_time_evolution_two_component(
        scenario=scenario,
        t_max_in_days=t_max_in_days,
        nu_start=1000,
        b_negative_times=b0_gs,
    )

    plot_time_evolution(
        t_in_days_sequence=t_in_days_list, R=R, FT_infty_sequence=FT_infty
    )
