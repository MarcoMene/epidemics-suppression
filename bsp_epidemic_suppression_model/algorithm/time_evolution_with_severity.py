from typing import List

from bsp_epidemic_suppression_model.algorithm.model_blocks import (
    compute_FA_from_FAs_and_previous_step_data,
    compute_FT_from_FA_and_DeltaAT,
    compute_r_R_components_from_FT,
)
from bsp_epidemic_suppression_model.utilities.model import FS
from bsp_epidemic_suppression_model.utilities.scenario import Scenario
from bsp_epidemic_suppression_model.utilities.functions_utils import (
    RealRange,
    list_from_f,
    f_from_list,
    round2,
    round2_list,
    ImproperProbabilityCumulativeFunction,
    integrate,
)


class StepData:
    def __init__(
            self,
            real_range: RealRange,
            t: float,  # Absolute time of this step
            papp: float,  # Probability that an infected at t has the app
            tildepapp: float,  # Probability that a source infected at t has the app
            tildepgs: List[
                float
            ],  # Probabilities that a source infected at t has each severity
            EtauC: float,  # Expected contagion time for source infected at t
            FT_infty: float,  # Probability that an infected at t tests positive
            FTapp_infty: float,  # Probability that an infected at t with the app tests positive
            FTnoapp_infty: float,  # Probability that an infected at t without the app tests positive
            tildeFTapp: ImproperProbabilityCumulativeFunction,
            # Distribution of testing time for source infected at t with app
            tildeFTnoapp: ImproperProbabilityCumulativeFunction,
            # Distribution of testing time for source infected at t with no app
            R: float,
            Rapp: float,
            Rnoapp: float,
    ):
        self.real_range = real_range
        self.t = t
        self.papp = papp
        self.tildepapp = tildepapp
        self.tildepgs = tildepgs
        self.EtauC = EtauC
        self.FT_infty = FT_infty
        self.FTapp_infty = FTapp_infty
        self.FTnoapp_infty = FTnoapp_infty
        self.tildeFTapp_values = list_from_f(f=tildeFTapp, real_range=real_range)
        self.tildeFTnoapp_values = list_from_f(
            f=tildeFTnoapp, real_range=real_range)
        self.R = R
        self.Rapp = Rapp
        self.Rnoapp = Rnoapp

    def tildeFTapp(self, tau):
        return f_from_list(f_values=self.tildeFTapp_values, real_range=self.real_range)(tau)

    def tildeFTnoapp(self, tau):
        return f_from_list(f_values=self.tildeFTnoapp_values, real_range=self.real_range)(tau)


def compute_time_evolution_with_severity(
        scenario: Scenario,
        real_range: RealRange,
        n_iterations: int = 6,
        verbose: bool = True,
):
    # ### INTERNAL UTILS ###
    tau_max = real_range.x_max

    # def f_from_list(f_values: list, tau) -> float:
    #     if tau < real_range.x_min:
    #         return f_values[0]
    #     if tau > real_range.x_max:
    #         return f_values[-1]
    #     i = int((tau - real_range.x_min) / real_range.step)
    #     return f_values[i]

    ### ITERATION

    step_data_list: List[StepData] = []

    for i in range(0, n_iterations):
        gs = range(scenario.n_severities)  # Values of severity G

        # Compute FAs components
        # FAsapp_ti_gs = [
        #     lambda tau: scenario.ssapp[0] * FS(tau),
        #     lambda tau: scenario.ssapp[1] * FS(tau),
        # ]
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
            real_range=real_range,
        )

        # Compute r, R components

        r0_ti_gs = [lambda tau, g=g: scenario.r0_gs[g](t_i, tau) for g in gs]
        (
            rapp_ti_gs,
            rnoapp_ti_gs,
            Rapp_ti_gs,
            Rnoapp_ti_gs,
        ) = compute_r_R_components_from_FT(
            FTapp_ti_gs=FTapp_ti_gs,
            FTnoapp_ti_gs=FTnoapp_ti_gs,
            r0_ti_gs=r0_ti_gs,
            xi=scenario.xi,
            tau_max=tau_max,
        )

        # Compute aggregate r (needed for EtauC), and R
        rapp_ti = lambda tau: sum(scenario.p_gs[g] * rapp_ti_gs[g](tau) for g in gs)
        rnoapp_ti = lambda tau: sum(scenario.p_gs[g] * rnoapp_ti_gs[g](tau) for g in gs)
        r_ti = lambda tau: scenario.papp(t_i) * rapp_ti(tau) + (
                1 - scenario.papp(t_i)
        ) * rnoapp_ti(tau)

        Rapp_ti = sum(scenario.p_gs[g] * Rapp_ti_gs[g] for g in gs)
        Rnoapp_ti = sum(scenario.p_gs[g] * Rnoapp_ti_gs[g] for g in gs)
        R_ti_gs = [
            scenario.papp(t_i) * Rapp_ti_gs[g]
            + (1 - scenario.papp(t_i)) * Rnoapp_ti_gs[g]
            for g in gs
        ]
        R_ti = scenario.papp(t_i) * Rapp_ti + (1 - scenario.papp(t_i)) * Rnoapp_ti

        # Compute source-based probabilities and distributions
        EtauC_ti = integrate(f=lambda tau: tau * r_ti(tau) / R_ti, a=0, b=tau_max)
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
                f" FAsapp_ti_gs(infty)={round2_list(FAsapp_ti_gs_infty)}\n"
                f" FAsnoapp_ti_gs(infty)={round2_list(FAsnoapp_ti_gs_infty)}\n"
                f" FAsapp_ti(infty)={round2(FAsapp_ti_infty)}\n"
                f" FAsnoapp_ti(infty)={round2(FAsnoapp_ti_infty)}\n"
                f" FAs_ti(infty)={round2(FAs_ti_infty)}\n"
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
            f" FAapp_ti_gs(infty)={round2_list(FAapp_ti_gs_infty)}\n"
            f" FAnoapp_ti_gs(infty)={round2_list(FAnoapp_ti_gs_infty)}\n"
            f" FAapp_ti(infty)={round2(FAapp_ti_infty)}\n"
            f" FAnoapp_ti(infty)={round2(FAnoapp_ti_infty)}\n"
            f" FA_ti(infty)={round2(FA_ti_infty)}\n"
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
            f" FTapp_ti_gs(infty)={round2_list(FTapp_ti_gs_infty)}\n"
            f" FTnoapp_ti_gs(infty)={round2_list(FTnoapp_ti_gs_infty)}\n"
            f" FTapp_ti(infty)={round2(FTapp_ti_infty)}\n"
            f" FTnoapp_ti(infty)={round2(FTnoapp_ti_infty)}\n"
            f" FT_ti(infty)={round2(FT_ti_infty)}\n"
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
