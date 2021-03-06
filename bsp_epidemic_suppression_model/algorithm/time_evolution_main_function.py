from typing import List

from bsp_epidemic_suppression_model.math_utilities.functions_utils import (
    RealRange,
    round2,
    round2_list,
    integrate,
)

from bsp_epidemic_suppression_model.model_utilities.epidemic_data import FS
from bsp_epidemic_suppression_model.model_utilities.scenario import Scenario

from bsp_epidemic_suppression_model.algorithm.model_blocks import (
    compute_FA_from_FAs_and_previous_step_data,
    compute_FT_from_FA_and_DeltaAT,
    compute_beta_and_R_components_from_FT,
)
from bsp_epidemic_suppression_model.algorithm.step_data import StepData


def compute_time_evolution(
    scenario: Scenario,
    real_range: RealRange,
    n_iterations: int = 6,
    verbose: bool = True,
) -> List[StepData]:
    """
    Given a Scenario, computes n_iterations steps of the algorithm, filling each time a StepData object and
    (if verbose=True) printing the relevant quantities computed.
    :param scenario: the Scenario object defining the input data of the mode.
    :param real_range: a RealRange object specifying the upper integration bound and the real numbers on which the
    functions and densities are sampled from one step to the next.
    :param n_iterations: the number of iterations.
    :param verbose: if True, the relevant quantities computed at each step are printed.
    :return: The list of StepData objects.
    """
    tau_max = real_range.x_max

    step_data_list: List[StepData] = []

    for i in range(0, n_iterations):
        gs = range(scenario.n_severities)  # Values of severity G

        # Compute FAs components
        FAsapp_ti_gs = [lambda tau, g=g: scenario.ssapp[g] * FS(tau) for g in gs]
        FAsnoapp_ti_gs = [lambda tau, g=g: scenario.ssnoapp[g] * FS(tau) for g in gs]

        # Compute FA components
        if i == 0:
            t_i = scenario.t_0
            FAapp_ti_gs = FAsapp_ti_gs
            FAnoapp_ti_gs = FAsnoapp_ti_gs
        else:
            previous_step_data = step_data_list[i - 1]
            t_i = previous_step_data.t + previous_step_data.EtauC
            FAapp_ti_gs, FAnoapp_ti_gs = compute_FA_from_FAs_and_previous_step_data(
                FAsapp_ti_gs=FAsapp_ti_gs,
                FAsnoapp_ti_gs=FAsnoapp_ti_gs,
                tildepapp_tim1=previous_step_data.tildepapp,
                tildeFTapp_tim1=previous_step_data.tildeFTapp,
                tildeFTnoapp_tim1=previous_step_data.tildeFTnoapp,
                EtauC_tim1=previous_step_data.EtauC,
                scapp=scenario.scapp,
                scnoapp=scenario.scnoapp,
            )

        # Compute FT components
        FTapp_ti_gs, FTnoapp_ti_gs = compute_FT_from_FA_and_DeltaAT(
            FAapp_ti_gs=FAapp_ti_gs,
            FAnoapp_ti_gs=FAnoapp_ti_gs,
            p_DeltaATapp=scenario.p_DeltaATapp,
            p_DeltaATnoapp=scenario.p_DeltaATnoapp,
        )

        # Compute beta, R components

        beta0_ti_gs = [lambda tau, g=g: scenario.beta0_gs[g](t_i, tau) for g in gs]
        (
            rapp_ti_gs,
            rnoapp_ti_gs,
            Rapp_ti_gs,
            Rnoapp_ti_gs,
        ) = compute_beta_and_R_components_from_FT(
            FTapp_ti_gs=FTapp_ti_gs,
            FTnoapp_ti_gs=FTnoapp_ti_gs,
            beta0_ti_gs=beta0_ti_gs,
            xi=scenario.xi,
            tau_max=tau_max,
        )

        # Compute aggregate beta (needed for EtauC), and R
        betaapp_ti = lambda tau: sum(scenario.p_gs[g] * rapp_ti_gs[g](tau) for g in gs)
        betanoapp_ti = lambda tau: sum(
            scenario.p_gs[g] * rnoapp_ti_gs[g](tau) for g in gs
        )
        beta_ti = lambda tau: scenario.papp(t_i) * betaapp_ti(tau) + (
            1 - scenario.papp(t_i)
        ) * betanoapp_ti(tau)

        Rapp_ti = sum(scenario.p_gs[g] * Rapp_ti_gs[g] for g in gs)
        Rnoapp_ti = sum(scenario.p_gs[g] * Rnoapp_ti_gs[g] for g in gs)
        R_ti_gs = [
            scenario.papp(t_i) * Rapp_ti_gs[g]
            + (1 - scenario.papp(t_i)) * Rnoapp_ti_gs[g]
            for g in gs
        ]
        R_ti = scenario.papp(t_i) * Rapp_ti + (1 - scenario.papp(t_i)) * Rnoapp_ti

        # Compute source-based probabilities and distributions
        EtauC_ti = integrate(f=lambda tau: tau * beta_ti(tau) / R_ti, a=0, b=tau_max)
        tildepapp_ti = scenario.papp(t_i) * Rapp_ti / R_ti
        tildep_ti_gs = [scenario.p_gs[g] * R_ti_gs[g] / R_ti for g in gs]
        tildeFTapp_ti = lambda tau: sum(
            tildep_ti_gs[g] * FTapp_ti_gs[g](tau) for g in gs
        )
        tildeFTnoapp_ti = lambda tau: sum(
            tildep_ti_gs[g] * FTnoapp_ti_gs[g](tau) for g in gs
        )

        # Limits and Recap
        step_recap = f"step {i}, t_i={round2(t_i)}\n"

        if i != 0:
            FAsapp_ti_gs_infty = [FAsapp_ti_gs[g](tau_max) for g in gs]
            FAsnoapp_ti_gs_infty = [FAsnoapp_ti_gs[g](tau_max) for g in gs]
            FAsapp_ti_infty = sum(scenario.p_gs[g] * FAsapp_ti_gs_infty[g] for g in gs)
            FAsnoapp_ti_infty = sum(
                scenario.p_gs[g] * FAsnoapp_ti_gs_infty[g] for g in gs
            )
            FAs_ti_infty = (
                scenario.papp(t_i) * FAsapp_ti_infty
                + (1 - scenario.papp(t_i)) * FAsnoapp_ti_infty
            )

            FAs_recap = (
                f" FAsapp_ti_gs(∞)={round2_list(FAsapp_ti_gs_infty)}\n"
                f" FAsnoapp_ti_gs(∞)={round2_list(FAsnoapp_ti_gs_infty)}\n"
                f" FAsapp_ti(∞)={round2(FAsapp_ti_infty)}\n"
                f" FAsnoapp_ti(∞)={round2(FAsnoapp_ti_infty)}\n"
                f" FAs_ti(∞)={round2(FAs_ti_infty)}\n"
            )
        else:
            FAs_recap = ""

        FAapp_ti_gs_infty = [FAapp_ti_gs[g](tau_max) for g in gs]
        FAnoapp_ti_gs_infty = [FAnoapp_ti_gs[g](tau_max) for g in gs]
        FAapp_ti_infty = sum(scenario.p_gs[g] * FAapp_ti_gs_infty[g] for g in gs)
        FAnoapp_ti_infty = sum(scenario.p_gs[g] * FAnoapp_ti_gs_infty[g] for g in gs)
        FA_ti_infty = (
            scenario.papp(t_i) * FAapp_ti_infty
            + (1 - scenario.papp(t_i)) * FAnoapp_ti_infty
        )

        FA_recap = (
            f" FAapp_ti_gs(∞)={round2_list(FAapp_ti_gs_infty)}\n"
            f" FAnoapp_ti_gs(∞)={round2_list(FAnoapp_ti_gs_infty)}\n"
            f" FAapp_ti(∞)={round2(FAapp_ti_infty)}\n"
            f" FAnoapp_ti(∞)={round2(FAnoapp_ti_infty)}\n"
            f" FA_ti(∞)={round2(FA_ti_infty)}\n"
        )

        FTapp_ti_gs_infty = [FTapp_ti_gs[g](tau_max) for g in gs]
        FTnoapp_ti_gs_infty = [FTnoapp_ti_gs[g](tau_max) for g in gs]
        FTapp_ti_infty = sum(scenario.p_gs[g] * FTapp_ti_gs_infty[g] for g in gs)
        FTnoapp_ti_infty = sum(scenario.p_gs[g] * FTnoapp_ti_gs_infty[g] for g in gs)
        FT_ti_infty = (
            scenario.papp(t_i) * FTapp_ti_infty
            + (1 - scenario.papp(t_i)) * FTnoapp_ti_infty
        )

        FT_recap = (
            f" FTapp_ti_gs(∞)={round2_list(FTapp_ti_gs_infty)}\n"
            f" FTnoapp_ti_gs(∞)={round2_list(FTnoapp_ti_gs_infty)}\n"
            f" FTapp_ti(∞)={round2(FTapp_ti_infty)}\n"
            f" FTnoapp_ti(∞)={round2(FTnoapp_ti_infty)}\n"
            f" FT_ti(∞)={round2(FT_ti_infty)}\n"
        )

        R_recap = (
            f" Rapp_ti_gs={round2_list(Rapp_ti_gs)}\n"
            f" Rnoapp_ti_gs={round2_list(Rnoapp_ti_gs)}\n"
            f" Rapp_ti={round2(Rapp_ti)}\n"
            f" Rnoapp_ti={round2(Rnoapp_ti)}\n"
            f" R_ti_gs={round2_list(R_ti_gs)}\n"
            f" R_ti={round2(R_ti)}\n"
        )

        other_recap = (
            f" papp_ti={round2(scenario.papp(t_i))}\n"
            f" tildepapp_ti={round2(tildepapp_ti)}\n"
            f" p_gs={round2_list(scenario.p_gs)}\n"
            f" tildep_ti_gs={round2_list(tildep_ti_gs)}\n"
            f" E(tauC_ti)={round2(EtauC_ti)} \n"
        )

        current_step_data = StepData(
            real_range=real_range,
            t=t_i,
            papp=scenario.papp(t_i),
            tildepapp=tildepapp_ti,
            tildepgs=tildep_ti_gs,
            EtauC=EtauC_ti,
            FT_infty=FT_ti_infty,
            FTapp_infty=FTapp_ti_infty,
            FTnoapp_infty=FTnoapp_ti_infty,
            tildeFTapp=tildeFTapp_ti,
            tildeFTnoapp=tildeFTnoapp_ti,
            R=R_ti,
            Rapp=Rapp_ti,
            Rnoapp=Rnoapp_ti,
        )

        step_data_list.append(current_step_data)

        if verbose:
            print(step_recap + FAs_recap + FA_recap + FT_recap + R_recap + other_recap)

    return step_data_list
