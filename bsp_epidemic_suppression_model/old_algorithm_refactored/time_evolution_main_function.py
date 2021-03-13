from typing import List

from bsp_epidemic_suppression_model.algorithm_refactored.model_blocks import (
    approximate_FA_from_FAs_and_previous_step_data,
    compute_aggregate_beta_t_two_components,
    compute_beta_and_R_components_from_FT,
)
from bsp_epidemic_suppression_model.algorithm_refactored.step_data import StepData
from bsp_epidemic_suppression_model.math_utilities.config import (
    TAU_MAX_IN_UNITS,
    TAU_UNIT_IN_DAYS,
)
from bsp_epidemic_suppression_model.math_utilities.discrete_distributions_utils import (
    generate_discrete_distribution_from_cdf_function,
)
from bsp_epidemic_suppression_model.math_utilities.general_utilities import round2
from bsp_epidemic_suppression_model.model_utilities.scenarios import (
    HomogeneousScenario,
    Scenario,
    homogeneous_to_two_components,
)
from bsp_epidemic_suppression_model.old_stuff.functions_utils import round2_sequence

_DEFAULT_N_ITERATIONS = 10


def compute_time_evolution_approximated_algorithm(
    scenario: Scenario, n_iterations: int = _DEFAULT_N_ITERATIONS, verbose: bool = True,
) -> List[StepData]:
    """
    Given a Scenario, computes n_iterations steps of the algorithm, filling each time a StepData
    object and (if verbose=True) printing the relevant quantities computed.
    :param scenario: the Scenario object defining the input data of the mode.
    :param n_iterations: the number of iterations.
    :param verbose: if True, the relevant quantities computed at each step are printed.
    :return: The list of StepData objects.
    """
    if isinstance(scenario, HomogeneousScenario):
        scenario = homogeneous_to_two_components(homogeneous_scenario=scenario)

    step_data_list: List[StepData] = []

    for i in range(0, n_iterations):
        gs = range(scenario.n_severities)  # Values of severity G

        # Compute tauA components
        if i == 0:
            t_i = scenario.t_0
            tauAsapp_ti_gs = [
                scenario.tauS.rescale_by_factor(scenario.ssapp[g](t_i)) for g in gs
            ]
            tauAsnoapp_ti_gs = [
                scenario.tauS.rescale_by_factor(scenario.ssnoapp[g](t_i)) for g in gs
            ]
            tauAapp_ti_gs = tauAsapp_ti_gs
            tauAnoapp_ti_gs = tauAsnoapp_ti_gs
        else:
            previous_step_data = step_data_list[i - 1]
            t_i = previous_step_data.t + previous_step_data.EtauC_in_days
            tauAsapp_ti_gs = [
                scenario.tauS.rescale_by_factor(scenario.ssapp[g](t_i)) for g in gs
            ]
            tauAsnoapp_ti_gs = [
                scenario.tauS.rescale_by_factor(scenario.ssnoapp[g](t_i)) for g in gs
            ]
            FAapp_ti_gs, FAnoapp_ti_gs = approximate_FA_from_FAs_and_previous_step_data(
                FAsapp_ti_gs=tuple(d.cdf for d in tauAsapp_ti_gs),
                FAsnoapp_ti_gs=tuple(d.cdf for d in tauAsnoapp_ti_gs),
                tildepapp_tim1=previous_step_data.tildepapp,
                tildeFTapp_tim1=previous_step_data.tildetauTapp.cdf,
                tildeFTnoapp_tim1=previous_step_data.tildetauTnoapp.cdf,
                EtauC_tim1_relative_tau_units=previous_step_data.EtauC_relative_time_units,
                scapp_ti=scenario.scapp(t_i),
                scnoapp_ti=scenario.scnoapp(t_i),
            )
            tauAapp_ti_gs = tuple(
                generate_discrete_distribution_from_cdf_function(
                    cdf=F, tau_min=0, tau_max=TAU_MAX_IN_UNITS,
                )
                for F in FAapp_ti_gs
            )
            tauAnoapp_ti_gs = tuple(
                generate_discrete_distribution_from_cdf_function(
                    cdf=F, tau_min=0, tau_max=TAU_MAX_IN_UNITS,
                )
                for F in FAnoapp_ti_gs
            )

        # Compute tauT components
        tauTapp_ti_gs = tuple(tauAapp_ti_gs[g] + scenario.DeltaATapp for g in gs)
        tauTnoapp_ti_gs = tuple(tauAnoapp_ti_gs[g] + scenario.DeltaATnoapp for g in gs)

        # Compute beta, R components

        (
            betaapp_ti_gs,
            betanoapp_ti_gs,
            Rapp_ti_gs,
            Rnoapp_ti_gs,
        ) = compute_beta_and_R_components_from_FT(
            tauTapp_ti_gs=tauTapp_ti_gs,
            tauTnoapp_ti_gs=tauTnoapp_ti_gs,
            beta0_ti_gs=scenario.b0_gs,
            xi_ti=scenario.xi(t_i),
        )

        # Compute aggregate beta (needed for EtauC), and R
        beta_ti = compute_aggregate_beta_t_two_components(
            p_gs=scenario.p_gs,
            papp_t=scenario.papp(t_i),
            betaapp_t_gs=betaapp_ti_gs,
            betanoapp_t_gs=betanoapp_ti_gs,
        )

        # betaapp_ti = lambda tau: sum(
        #     scenario.p_gs[g] * betaapp_ti_gs[g].pmf(tau) for g in gs
        # )
        # betanoapp_ti = lambda tau: sum(
        #     scenario.p_gs[g] * betanoapp_ti_gs[g].pmf(tau) for g in gs
        # )
        # beta_ti = lambda tau: scenario.papp(t_i) * betaapp_ti(tau) + (
        #     1 - scenario.papp(t_i)
        # ) * betanoapp_ti(tau)

        Rapp_ti = sum(scenario.p_gs[g] * Rapp_ti_gs[g] for g in gs)
        Rnoapp_ti = sum(scenario.p_gs[g] * Rnoapp_ti_gs[g] for g in gs)
        R_ti_gs = [
            scenario.papp(t_i) * Rapp_ti_gs[g]
            + (1 - scenario.papp(t_i)) * Rnoapp_ti_gs[g]
            for g in gs
        ]
        R_ti = scenario.papp(t_i) * Rapp_ti + (1 - scenario.papp(t_i)) * Rnoapp_ti

        # Compute source-based probabilities and distributions
        tauC_ti = beta_ti.normalize()
        EtauC_ti_relative_time_units = tauC_ti.mean()
        EtauC_ti_days = EtauC_ti_relative_time_units * TAU_UNIT_IN_DAYS

        tildepapp_ti = scenario.papp(t_i) * Rapp_ti / R_ti
        tildep_ti_gs = [scenario.p_gs[g] * R_ti_gs[g] / R_ti for g in gs]
        # tildeFTapp_ti = lambda tau: sum(
        #     tildep_ti_gs[g] * tauTapp_ti_gs[g].cdf(tau) for g in gs
        # )
        tildetauTapp_ti = generate_discrete_distribution_from_cdf_function(
            cdf=lambda tau: sum(
                tildep_ti_gs[g] * tauTapp_ti_gs[g].cdf(tau) for g in gs
            ),
            tau_min=1,
            tau_max=TAU_MAX_IN_UNITS,
        )
        # tildeFTnoapp_ti = lambda tau: sum(
        #     tildep_ti_gs[g] * tauTnoapp_ti_gs[g].cdf(tau) for g in gs
        # )
        tildetauTnoapp_ti = generate_discrete_distribution_from_cdf_function(
            cdf=lambda tau: sum(
                tildep_ti_gs[g] * tauTnoapp_ti_gs[g].cdf(tau) for g in gs
            ),
            tau_min=1,
            tau_max=TAU_MAX_IN_UNITS,
        )

        # Limits and Recap
        step_recap = f"step {i}, t_i={round2(t_i)}\n"

        if i != 0:
            FAsapp_ti_gs_infty = [tauAsapp_ti_gs[g].total_mass for g in gs]
            FAsnoapp_ti_gs_infty = [tauAsnoapp_ti_gs[g].total_mass for g in gs]
            FAsapp_ti_infty = sum(scenario.p_gs[g] * FAsapp_ti_gs_infty[g] for g in gs)
            FAsnoapp_ti_infty = sum(
                scenario.p_gs[g] * FAsnoapp_ti_gs_infty[g] for g in gs
            )
            FAs_ti_infty = (
                scenario.papp(t_i) * FAsapp_ti_infty
                + (1 - scenario.papp(t_i)) * FAsnoapp_ti_infty
            )

            FAs_recap = (
                f" FAsapp_ti_gs(∞)={round2_sequence(FAsapp_ti_gs_infty)}\n"
                f" FAsnoapp_ti_gs(∞)={round2_sequence(FAsnoapp_ti_gs_infty)}\n"
                f" FAsapp_ti(∞)={round2(FAsapp_ti_infty)}\n"
                f" FAsnoapp_ti(∞)={round2(FAsnoapp_ti_infty)}\n"
                f" FAs_ti(∞)={round2(FAs_ti_infty)}\n"
            )
        else:
            FAs_recap = ""

        FAapp_ti_gs_infty = [tauAapp_ti_gs[g].total_mass for g in gs]
        FAnoapp_ti_gs_infty = [tauAnoapp_ti_gs[g].total_mass for g in gs]
        FAapp_ti_infty = sum(scenario.p_gs[g] * FAapp_ti_gs_infty[g] for g in gs)
        FAnoapp_ti_infty = sum(scenario.p_gs[g] * FAnoapp_ti_gs_infty[g] for g in gs)
        FA_ti_infty = (
            scenario.papp(t_i) * FAapp_ti_infty
            + (1 - scenario.papp(t_i)) * FAnoapp_ti_infty
        )

        FA_recap = (
            f" FAapp_ti_gs(∞)={round2_sequence(FAapp_ti_gs_infty)}\n"
            f" FAnoapp_ti_gs(∞)={round2_sequence(FAnoapp_ti_gs_infty)}\n"
            f" FAapp_ti(∞)={round2(FAapp_ti_infty)}\n"
            f" FAnoapp_ti(∞)={round2(FAnoapp_ti_infty)}\n"
            f" FA_ti(∞)={round2(FA_ti_infty)}\n"
        )

        FTapp_ti_gs_infty = [tauTapp_ti_gs[g].total_mass for g in gs]
        FTnoapp_ti_gs_infty = [tauTnoapp_ti_gs[g].total_mass for g in gs]
        FTapp_ti_infty = sum(scenario.p_gs[g] * FTapp_ti_gs_infty[g] for g in gs)
        FTnoapp_ti_infty = sum(scenario.p_gs[g] * FTnoapp_ti_gs_infty[g] for g in gs)
        FT_ti_infty = (
            scenario.papp(t_i) * FTapp_ti_infty
            + (1 - scenario.papp(t_i)) * FTnoapp_ti_infty
        )

        FT_recap = (
            f" FTapp_ti_gs(∞)={round2_sequence(FTapp_ti_gs_infty)}\n"
            f" FTnoapp_ti_gs(∞)={round2_sequence(FTnoapp_ti_gs_infty)}\n"
            f" FTapp_ti(∞)={round2(FTapp_ti_infty)}\n"
            f" FTnoapp_ti(∞)={round2(FTnoapp_ti_infty)}\n"
            f" FT_ti(∞)={round2(FT_ti_infty)}\n"
        )

        R_recap = (
            f" Rapp_ti_gs={round2_sequence(Rapp_ti_gs)}\n"
            f" Rnoapp_ti_gs={round2_sequence(Rnoapp_ti_gs)}\n"
            f" Rapp_ti={round2(Rapp_ti)}\n"
            f" Rnoapp_ti={round2(Rnoapp_ti)}\n"
            f" R_ti_gs={round2_sequence(R_ti_gs)}\n"
            f" R_ti={round2(R_ti)}\n"
        )

        other_recap = (
            f" papp_ti={round2(scenario.papp(t_i))}\n"
            f" tildepapp_ti={round2(tildepapp_ti)}\n"
            f" p_gs={round2_sequence(scenario.p_gs)}\n"
            f" tildep_ti_gs={round2_sequence(tildep_ti_gs)}\n"
            f" E(tauC_ti)={round2(EtauC_ti_days)} \n"
        )

        current_step_data = StepData(
            t=t_i,
            papp=scenario.papp(t_i),
            tildepapp=tildepapp_ti,
            tildepgs=tildep_ti_gs,
            EtauC_relative_time_units=EtauC_ti_relative_time_units,
            EtauC_in_days=EtauC_ti_days,
            FT_infty=FT_ti_infty,
            FTapp_infty=FTapp_ti_infty,
            FTnoapp_infty=FTnoapp_ti_infty,
            tildetauTapp=tildetauTapp_ti,
            tildetauTnoapp=tildetauTnoapp_ti,
            R=R_ti,
            Rapp=Rapp_ti,
            Rnoapp=Rnoapp_ti,
        )

        step_data_list.append(current_step_data)

        if verbose:
            print(step_recap + FAs_recap + FA_recap + FT_recap + R_recap + other_recap)

    return step_data_list
