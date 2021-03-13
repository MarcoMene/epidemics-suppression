from epidemic_suppression_algorithms.homogeneous_evolution_algorithm import (
    compute_time_evolution_homogeneous_case,
)
from examples.plotting_utils import plot_homogeneous_time_evolution
from math_utilities.discrete_distributions_utils import (
    DiscreteDistributionOnNonNegatives,
    delta_distribution,
)
from model_utilities.epidemic_data import (
    R0,
    make_scenario_parameters_for_asymptomatic_symptomatic_model,
    tauS,
)
from model_utilities.scenarios import HomogeneousScenario

p_gs, b0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()


def time_evolution_homogeneous_model_optimistic_scenario_example():
    """
    Example in which there is no app usage, and the sensitivities s^S and s^C are quite high
    """
    t_max_in_days = 20

    DeltaAT_in_days = 2

    t_0 = 0

    scenario = HomogeneousScenario(
        p_gs=p_gs,
        b0_gs=b0_gs,
        tauS=tauS,
        t_0=0,
        ss=(lambda t: 0, lambda t: 0.5 if t >= t_0 else 0),
        sc=lambda t: 0.7 if t >= t_0 else 0,
        xi=lambda t: 0.9 if t >= t_0 else 0,
        DeltaAT=delta_distribution(peak_tau_in_days=DeltaAT_in_days),
    )

    (
        t_in_days_list,
        nu,
        nu0,
        R,
        R_by_severity,
        FT_infty,
    ) = compute_time_evolution_homogeneous_case(
        scenario=scenario,
        t_max_in_days=t_max_in_days,
        nu_start=1000,
        b_negative_times=b0_gs,
        threshold_to_stop=0.001,
    )

    plot_homogeneous_time_evolution(
        t_in_days_sequence=t_in_days_list,
        R_ts=R,
        R0=R0,
        FT_infty_sequence=FT_infty,
        nu_ts=nu,
        nu0_ts=nu0,
    )


def time_evolution_homogeneous_model_optimistic_scenario_schematic_data_example():
    """
    Example in which there is no app usage, and the sensitivities s^S and s^C are quite high
    """
    t_max_in_days = 20

    DeltaAT_in_days = 2

    t_0 = 0

    def schematize(
        d: DiscreteDistributionOnNonNegatives,
    ) -> DiscreteDistributionOnNonNegatives:
        return DiscreteDistributionOnNonNegatives(
            pmf_values=[tauS.total_mass], tau_min=int(tauS.mean()), improper=d._improper
        )

    tauS_schematic = schematize(d=tauS)
    b0_gs_schematic = tuple(schematize(d=b0_g) for b0_g in b0_gs)

    scenario = HomogeneousScenario(
        p_gs=p_gs,
        b0_gs=b0_gs_schematic,
        tauS=tauS_schematic,
        t_0=0,
        ss=(lambda t: 0, lambda t: 0.5 if t >= t_0 else 0),
        sc=lambda t: 0.7 if t >= t_0 else 0,
        xi=lambda t: 0.9 if t >= t_0 else 0,
        DeltaAT=delta_distribution(peak_tau_in_days=DeltaAT_in_days),
    )

    (
        t_in_days_list,
        nu,
        nu0,
        R,
        R_by_severity,
        FT_infty,
    ) = compute_time_evolution_homogeneous_case(
        scenario=scenario,
        t_max_in_days=t_max_in_days,
        nu_start=1000,
        b_negative_times=b0_gs,
    )

    plot_homogeneous_time_evolution(
        t_in_days_sequence=t_in_days_list,
        R_ts=R,
        R0=R0,
        FT_infty_sequence=FT_infty,
        nu_ts=nu,
        nu0_ts=nu0,
    )


def time_evolution_homogeneous_model_no_measures_example():
    """
    Example in which there is no app usage, and the sensitivities s^S and s^C are quite high
    """
    t_max_in_days = 20

    DeltaAT_in_days = 2

    t_0 = 0

    scenario = HomogeneousScenario(
        p_gs=p_gs,
        b0_gs=tuple(b0_g.rescale_by_factor(scale_factor=0.9) for b0_g in b0_gs),
        tauS=tauS,
        t_0=t_0,
        ss=(lambda t: 0, lambda t: 0),
        sc=lambda t: 0,
        xi=lambda t: 0,
        DeltaAT=delta_distribution(peak_tau_in_days=DeltaAT_in_days),
    )

    (
        t_in_days_list,
        nu,
        nu0,
        R,
        R_by_severity,
        FT_infty,
    ) = compute_time_evolution_homogeneous_case(
        scenario=scenario,
        t_max_in_days=t_max_in_days,
        nu_start=1000,
        b_negative_times=b0_gs,
    )

    plot_homogeneous_time_evolution(
        t_in_days_sequence=t_in_days_list,
        R_ts=R,
        R0=R0,
        FT_infty_sequence=FT_infty,
        nu_ts=nu,
        nu0_ts=nu0,
    )
