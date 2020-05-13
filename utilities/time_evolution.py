from utilities.model import r0, suppressed_r_from_test_cdf, R0, FS, effectiveness_from_R
from utilities.scenario import Scenario
from utilities.utils import RealRange, DeltaMeasure, f_from_list, list_from_f, convolve, round2
import scipy.integrate as sci_integrate
from dataclasses import dataclass

import warnings

warnings.filterwarnings('ignore')

tau_max = 30
integration_step = 0.05


@dataclass
class TimeEvolutionResult:
    """
    Result of a time evolution calculation.
    """
    t_list: list
    FTinfty_list: list
    FTinftyapp_list: list
    FTinftynoapp_list: list
    R_list: list
    Rapp_list: list
    Rnoapp_list: list
    P_list: list
    tauC_exp_list: list

    @property
    def last_t(self):
        return self.t_list[-1]

    @property
    def last_FTinfty(self):
        return self.FTinfty_list[-1]

    @property
    def last_FTinftyapp(self):
        return self.FTinftyapp_list[-1]

    @property
    def last_FTinftynoapp(self):
        return self.FTinftynoapp_list[-1]

    @property
    def last_R(self):
        return self.R_list[-1]

    @property
    def last_Rapp(self):
        return self.Rapp_list[-1]

    @property
    def last_Rnoapp(self):
        return self.Rnoapp_list[-1]

    @property
    def last_P(self):
        return self.P_list[-1]

    @property
    def last_tauC_exp(self):
        return self.tauC_exp_list[-1]


def integrate(f: callable, a: float = 0, b: float = tau_max) -> float:
    """
    Integral of f from a to b
    """
    return sci_integrate.quad(f, a, b)[0]


def compute_time_evolution(scenario: Scenario,
                           n_iterations: int = 6,
                           verbose: bool = False):
    """
    Computes the time evolution o R_t and other quantities.
    """

    def prev_value(a_list):
        if a_list is None or len(a_list) < 1:
            raise ValueError("provide a non-empty list")
        return a_list[-1]

    t_list = []
    tildepC_list = []
    FTapp_list = []
    FTnoapp_list = []
    FTinftyapp_list = []
    FTinftynoapp_list = []
    FTinfty_list = []
    R_list = []
    Rapp_list = []
    Rnoapp_list = []
    P_list = []
    tauC_exp_list = []

    for i in range(0, n_iterations):

        if i == 0:
            t = 0
            FAapp_t = scenario.FAsapp
            FAnoapp_t = scenario.FAsnoapp

        else:

            FTapp_prev_values = prev_value(FTapp_list)
            FTapp_prev = f_from_list(FTapp_prev_values, RealRange(0, tau_max, integration_step))

            FTnoapp_prev_values = prev_value(FTnoapp_list)
            FTnoapp_prev = f_from_list(FTnoapp_prev_values, RealRange(0, tau_max, integration_step))

            tildepC_prev_values = prev_value(tildepC_list)
            tildepC_prev = f_from_list(tildepC_prev_values, RealRange(0, tau_max, integration_step))

            tildetauC_exp = integrate(lambda tau: tau * tildepC_prev(tau))
            t = prev_value(t_list) + tildetauC_exp

            P_prev = prev_value(P_list)

            # Evolution equations here:
            LHSapp = lambda tau: scenario.sCapp * P_prev * FTapp_prev(tau) + (
                        1 - P_prev) * scenario.sCnoapp * FTnoapp_prev(tau)
            FAcapp_t = lambda tau: LHSapp(tau + tildetauC_exp)
            FAapp_t = lambda tau: scenario.FAsapp(tau) + FAcapp_t(tau) - scenario.FAsapp(tau) * FAcapp_t(tau)

            LHSnoapp = lambda tau: scenario.sCnoapp * P_prev * FTapp_prev(tau) + (
                        1 - P_prev) * scenario.sCnoapp * FTnoapp_prev(tau)
            FAcnoapp_t = lambda tau: LHSnoapp(tau + tildetauC_exp)
            FAnoapp_t = lambda tau: scenario.FAsnoapp(tau) + FAcnoapp_t(tau) - scenario.FAsnoapp(tau) * FAcnoapp_t(tau)

        FTapp_t = convolve(FAapp_t, scenario.p_DeltaATapp, RealRange(0, tau_max, integration_step))
        rapp_t = suppressed_r_from_test_cdf(lambda tau: r0(tau), FTapp_t, scenario.xi)

        FTnoapp_t = convolve(FAnoapp_t, scenario.p_DeltaATnoapp, RealRange(0, tau_max, integration_step))
        rnoapp_t = suppressed_r_from_test_cdf(lambda tau: r0(tau), FTnoapp_t, scenario.xi)

        r_t = lambda tau: scenario.epsilon(t) * rapp_t(tau) + (1 - scenario.epsilon(t)) * rnoapp_t(tau)

        Rapp_t = integrate(rapp_t)
        Rnoapp_t = integrate(rnoapp_t)
        R_t = integrate(r_t)

        tauC_exp = integrate(lambda tau: tau * r_t(tau) / R_t)

        P_t = scenario.epsilon(t) * Rapp_t / R_t

        tilder_t = lambda tau: P_t * rapp_t(tau) + (1 - P_t) * rnoapp_t(tau)
        tildeR_t = integrate(tilder_t)
        tildepC_t = lambda tau: tilder_t(tau) / tildeR_t

        FTinftyapp = FTapp_t(tau_max)
        FTinftynoapp = FTnoapp_t(tau_max)
        FTinfty = scenario.epsilon(t) * FTinftyapp + (1 - scenario.epsilon(t)) * FTinftynoapp

        t_list.append(t)
        r_t_values = list_from_f(r_t, RealRange(0, tau_max, integration_step))
        tildepC_t_values = list_from_f(tildepC_t, RealRange(0, tau_max, integration_step))
        FTapp_t_values = list_from_f(FTapp_t, RealRange(0, tau_max, integration_step))
        FTnoapp_t_values = list_from_f(FTnoapp_t, RealRange(0, tau_max, integration_step))
        tildepC_list.append(tildepC_t_values)
        FTapp_list.append(FTapp_t_values)
        FTnoapp_list.append(FTnoapp_t_values)
        FTinftyapp_list.append(FTinftyapp)
        FTinftynoapp_list.append(FTinftynoapp)
        FTinfty_list.append(FTinfty)
        R_list.append(R_t)
        Rapp_list.append(Rapp_t)
        Rnoapp_list.append(Rnoapp_t)
        P_list.append(P_t)
        tauC_exp_list.append(tauC_exp)

        if verbose:
            print(f"t={round2(t)} \n"
                  f" FTapp_t(infty)={round2(FTinftyapp)} \n"
                  f" FTnoapp_t(infty)={round2(FTinftynoapp)} \n"
                  f" FT_t(infty)={round2(FTinfty)} \n"
                  f" epsilon_t={scenario.epsilon(t)} \n" 
                  f" P_t={round2(P_t)} \n"
                  f" R_t={round2(R_t)} \n"
                  f" E(tauC)={round2(tauC_exp)} \n"
                  )
    return TimeEvolutionResult(
        t_list=t_list,
        FTinfty_list=FTinfty_list,
        FTinftyapp_list=FTinftyapp_list,
        FTinftynoapp_list=FTinftynoapp_list,
        R_list=R_list,
        Rapp_list=Rapp_list,
        Rnoapp_list=Rnoapp_list,
        P_list=P_list,
        tauC_exp_list=tauC_exp_list
    )
