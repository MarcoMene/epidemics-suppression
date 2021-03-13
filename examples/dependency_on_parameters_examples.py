import math

import numpy as np
from matplotlib import pyplot as plt

from epidemic_suppression_algorithms.evolution_with_app_algorithm import (
    compute_time_evolution_with_app,
)
from epidemic_suppression_algorithms.homogeneous_evolution_algorithm import (
    compute_time_evolution_homogeneous_case,
)
from math_utilities.config import TAU_MAX_IN_UNITS, TAU_UNIT_IN_DAYS
from math_utilities.discrete_distributions_utils import (
    delta_distribution,
    generate_discrete_distribution_from_pdf_function,
)
from math_utilities.general_utilities import effectiveness
from model_utilities.epidemic_data import (
    R0,
    make_scenario_parameters_for_asymptomatic_symptomatic_model,
    rho0,
    tauS,
)
from model_utilities.scenarios import HomogeneousScenario, ScenarioWithApp


def dependency_on_testing_timeliness_homogeneous_model_example():
    """
    Example of several computations of the limit Eff_∞ in homogeneous scenarios
    in which the time interval Δ^{A → T} varies from 0 to 10 days.
    """
    # Severities: gs = [asymptomatic, symptomatic]
    p_gs, b0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()
    t_0 = 0

    DeltaAT_values_list = [i for i in range(10)]
    Effinfty_values_list = []

    for DeltaAT_in_days in DeltaAT_values_list:

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
            t_max_in_days=20,
            nu_start=1000,
            b_negative_times=b0_gs,
            verbose=False,
            threshold_to_stop=0.001,
        )

        Rinfty = R[-1]
        Effinfty = effectiveness(Rinfty, R0)
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
    Example of several computations of the limit Eff_∞ in homogeneous scenarios
    in which the fraction p_sym of symptomatic individuals  and their contribution to R^0 vary.
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

        # Severities: gs = (asymptomatic, symptomatic)
        p_gs, b0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model(
            p_sym=p_sym,
            contribution_of_symptomatics_to_R0=contribution_of_symptomatics_to_R0,
        )

        scenario = HomogeneousScenario(
            p_gs=p_gs,
            b0_gs=b0_gs,
            tauS=tauS,
            t_0=0,
            ss=(lambda t: 0, lambda t: 0.5),
            sc=lambda t: 0.7,
            xi=lambda t: 0.9,
            DeltaAT=delta_distribution(peak_tau_in_days=2),
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
            t_max_in_days=20,
            nu_start=1000,
            b_negative_times=b0_gs,
            verbose=False,
            threshold_to_stop=0.001,
        )

        Rinfty = R[-1]
        Effinfty = effectiveness(Rinfty, R0)
        Effinfty_values_list.append(Effinfty)

        x_axis_list.append(
            f"p_sym = {p_sym},\nκ={round(contribution_of_symptomatics_to_R0, 2)},\nR^0_sym={round(R0_sym, 2)},\nR^0_asy={round(R0_asy, 2)}"
        )

    plt.xticks(p_sym_list, x_axis_list, rotation=0)
    plt.xlim(0, p_sym_list[-1])
    plt.ylim(0, 0.8)
    plt.ylabel("Eff_∞")
    plt.grid(True)
    plt.plot(p_sym_list, Effinfty_values_list, color="black",),

    plt.show()


def dependency_on_infectiousness_width_homogeneous_model_example():
    """
    Example of several computations of the limit Eff_∞ in homogeneous scenarios
    in which the default distribution ρ^0 of the generation time is rescaled by different factors.
    """

    infectiousness_rescale_factors = [0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8]

    expected_default_generation_times_list = []
    Effinfty_values_list = []

    for f in infectiousness_rescale_factors:

        def rescaled_rho0_function(tau):
            return (1 / f) * rho0(tau / f)

        rescaled_rho0 = generate_discrete_distribution_from_pdf_function(
            pdf=lambda tau: rescaled_rho0_function(tau * TAU_UNIT_IN_DAYS)
            * TAU_UNIT_IN_DAYS,
            tau_min=1,
            tau_max=TAU_MAX_IN_UNITS,
            normalize=True,
        )
        EtauC0 = rescaled_rho0.mean()  # Expected default generation time
        expected_default_generation_times_list.append(EtauC0)

        # Severities: gs = (asymptomatic, symptomatic)
        p_gs, b0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model(
            rho0_discrete=rescaled_rho0
        )

        scenario = HomogeneousScenario(
            p_gs=p_gs,
            b0_gs=b0_gs,
            tauS=tauS,
            t_0=0,
            ss=(lambda t: 0, lambda t: 0.5),
            sc=lambda t: 0.7,
            xi=lambda t: 0.9,
            DeltaAT=delta_distribution(peak_tau_in_days=2),
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
            t_max_in_days=20,
            nu_start=1000,
            b_negative_times=b0_gs,
            verbose=False,
            threshold_to_stop=0.001,
        )

        Rinfty = R[-1]
        Effinfty = effectiveness(Rinfty, R0)
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
    Example of several computations of the limit Eff_∞ with app usage,
    where the parameters s^{s,app} and s^{c,app} vary.
    """

    # Severities: gs = (asymptomatic, symptomatic)
    p_gs, b0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()
    ssapp_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    scapp_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    Effinfty_values_list = []

    for scapp in scapp_list:
        Effinfty_values_list_for_scapp = []
        for ssapp in ssapp_list:
            scenario = ScenarioWithApp(
                p_gs=p_gs,
                b0_gs=b0_gs,
                tauS=tauS,
                t_0=0,
                ssapp=(lambda t: 0, lambda t: ssapp),
                ssnoapp=(lambda t: 0, lambda t: 0.2),
                scapp=lambda t: scapp,
                scnoapp=lambda t: 0.2,
                xi=lambda t: 0.9,
                DeltaATapp=delta_distribution(peak_tau_in_days=2),
                DeltaATnoapp=delta_distribution(peak_tau_in_days=4),
                papp=lambda t: 0.6,
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
                t_max_in_days=20,
                nu_start=1000,
                b_negative_times=b0_gs,
                threshold_to_stop=0.001,
            )

            Rinfty = R[-1]
            Effinfty = effectiveness(Rinfty, R0)
            Effinfty_values_list_for_scapp.append(Effinfty)

            print(
                f"s^{{s,app}} = {ssapp}, s^{{c,app}} = {scapp}, Eff_∞ = {round(Effinfty, 2)}"
            )

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
    # Severities: gs = (asymptomatic, symptomatic)
    p_gs, b0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()

    papp_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    Effinfty_values_list = []

    for papp in papp_list:

        scenario = ScenarioWithApp(
            p_gs=p_gs,
            b0_gs=b0_gs,
            tauS=tauS,
            t_0=0,
            ssapp=(lambda t: 0, lambda t: 0.5),
            ssnoapp=(lambda t: 0, lambda t: 0.2),
            scapp=lambda t: 0.7,
            scnoapp=lambda t: 0.2,
            xi=lambda t: 0.9,
            DeltaATapp=delta_distribution(peak_tau_in_days=2),
            DeltaATnoapp=delta_distribution(peak_tau_in_days=4),
            papp=lambda t: papp,
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
            t_max_in_days=40,
            nu_start=1000,
            b_negative_times=b0_gs,
            threshold_to_stop=0.001,
        )

        Rinfty = R[-1]
        Effinfty = effectiveness(Rinfty, R0)
        Effinfty_values_list.append(Effinfty)

    fig = plt.figure(figsize=(10, 15))

    Rinfty_plot = fig.add_subplot(111)
    Rinfty_plot.set_xlabel("p^app")
    Rinfty_plot.set_ylabel("Eff_∞")
    Rinfty_plot.grid(True)
    Rinfty_plot.set_xlim(0, 1)
    Rinfty_plot.set_ylim(0, 0.3)
    Rinfty_plot.plot(papp_list, Effinfty_values_list, color="black",),

    plt.show()
