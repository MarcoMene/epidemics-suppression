from epidemic_suppression_algorithms.evolution_with_app_algorithm import (
    compute_time_evolution_with_app,
)
from examples.plotting_utils import plot_time_evolution_with_app
from math_utilities.discrete_distributions_utils import delta_distribution
from model_utilities.epidemic_data import (
    R0,
    make_scenario_parameters_for_asymptomatic_symptomatic_model,
    tauS,
)
from model_utilities.scenarios import ScenarioWithApp

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
        DeltaATapp=delta_distribution(peak_tau_in_days=DeltaAT_in_days),
        DeltaATnoapp=delta_distribution(peak_tau_in_days=DeltaAT_in_days),
        epsilon_app=lambda t: 0.5,
    )

    (
        t_in_days_list,
        nu,
        nu0,
        Fsigmaapp_infty,
        R,
        R_app,
        R_noapp,
        FT_infty,
        FT_app_infty,
        FT_noapp_infty,
    ) = compute_time_evolution_with_app(
        scenario=scenario,
        t_max_in_days=t_max_in_days,
        nu_start=1000,
        b_negative_times=b0_gs,
    )

    plot_time_evolution_with_app(
        t_in_days_sequence=t_in_days_list,
        epsilon_app=scenario.epsilon_app,
        Fsigmaapp_infty=Fsigmaapp_infty,
        R_ts=R,
        Rapp_ts=R_app,
        Rnoapp_ts=R_noapp,
        R0=R0,
        FT_ts_infty=FT_infty,
        FT_ts_app_infty=FT_app_infty,
        FT_ts_noapp_infty=FT_noapp_infty,
        nu_ts=nu,
        nu0_ts=nu0,
    )


def time_evolution_two_component_model_optimistic_scenario_example2():
    """
    Example in which there is no app usage, and the sensitivities s^S and s^C are quite high
    """
    t_max_in_days = 20

    DeltaAT_app_in_days = 2
    DeltaAT_noapp_in_days = 2

    t_0 = 0

    scenario = ScenarioWithApp(
        p_gs=p_gs,
        b0_gs=b0_gs,
        tauS=tauS,
        t_0=0,
        ssapp=(lambda t: 0, lambda t: 0.7 if t >= t_0 else 0),
        ssnoapp=(lambda t: 0, lambda t: 0.5 if t >= t_0 else 0),
        scapp=lambda t: 0.7 if t >= t_0 else 0,
        scnoapp=lambda t: 0.3 if t >= t_0 else 0,
        xi=lambda t: 0.9 if t >= t_0 else 0,
        DeltaATapp=delta_distribution(peak_tau_in_days=DeltaAT_app_in_days),
        DeltaATnoapp=delta_distribution(peak_tau_in_days=DeltaAT_noapp_in_days),
        epsilon_app=lambda t: 0.05 * t if t < 10 else 0.5,
    )

    (
        t_in_days_list,
        nu,
        nu0,
        Fsigmaapp_infty,
        R,
        R_app,
        R_noapp,
        FT_infty,
        FT_app_infty,
        FT_noapp_infty,
    ) = compute_time_evolution_with_app(
        scenario=scenario,
        t_max_in_days=t_max_in_days,
        nu_start=1000,
        b_negative_times=b0_gs,
    )

    plot_time_evolution_with_app(
        t_in_days_sequence=t_in_days_list,
        epsilon_app=scenario.epsilon_app,
        Fsigmaapp_infty=Fsigmaapp_infty,
        R_ts=R,
        Rapp_ts=R_app,
        Rnoapp_ts=R_noapp,
        R0=R0,
        FT_ts_infty=FT_infty,
        FT_ts_app_infty=FT_app_infty,
        FT_ts_noapp_infty=FT_noapp_infty,
        nu_ts=nu,
        nu0_ts=nu0,
    )
