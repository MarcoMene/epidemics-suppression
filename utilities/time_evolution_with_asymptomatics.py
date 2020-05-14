import warnings

from utilities.model import r0sy, r0asy, suppressed_r_from_test_cdf, alpha
from utilities.scenario import Scenario
from utilities.time_evolution import TimeEvolutionResult, tau_max, integration_step, integrate
from utilities.utils import RealRange, f_from_list, list_from_f, convolve, round2

warnings.filterwarnings('ignore')


def compute_time_evolution_with_asymptomatics(scenario: Scenario,
                                              n_iterations: int = 6,
                                              verbose: bool = False):
    """
    Computes the time evolution of R_t and other quantities.
    """

    def prev_value(a_list):
        if a_list is None or len(a_list) < 1:
            raise ValueError("provide a non-empty list")
        return a_list[-1]

    def previous_function(a_list):
        return f_from_list(prev_value(a_list), RealRange(0, tau_max, integration_step))

    t_list = []
    tildepC_list = []
    FTappsy_list = []
    FTappasy_list = []
    FTnoappsy_list = []
    FTnoappasy_list = []
    FTinftyapp_list = []
    FTinftynoapp_list = []
    FTinfty_list = []
    R_list = []
    Rapp_list = []
    Rnoapp_list = []
    P_list = []
    Psy_list = []
    tauC_exp_list = []

    for i in range(0, n_iterations):

        if i == 0:
            t = 0
            FAappsy_t = scenario.FAsapp
            FAappasy_t = lambda tau: 0
            FAnoappsy_t = scenario.FAsnoapp
            FAnoappasy_t = lambda tau: 0

        else:

            FTappsy_prev = previous_function(FTappsy_list)
            FTappasy_prev = previous_function(FTappasy_list)
            FTnoappsy_prev = previous_function(FTnoappsy_list)
            FTnoappasy_prev = previous_function(FTnoappasy_list)

            tildepC_prev = previous_function(tildepC_list)

            tildetauC_exp = integrate(lambda tau: tau * tildepC_prev(tau))
            t = prev_value(t_list) + tildetauC_exp

            P_prev = prev_value(P_list)
            Psy_prev = prev_value(Psy_list)

            # Evolution equations here:
            def evolution(FN_source: callable, FAs_recipient, time_step):
                FAc_recipient = lambda tau: FN_source(tau + time_step)
                FA_recipient = lambda tau: FAs_recipient(tau) + FAc_recipient(tau) - FAs_recipient(tau) * FAc_recipient(
                    tau)
                return FA_recipient

            FTapp_prev = lambda tau: Psy_prev * FTappsy_prev(tau) * (1 - Psy_prev) * FTappasy_prev(tau)
            FTnoapp_prev = lambda tau: Psy_prev * FTnoappsy_prev(tau) * (1 - Psy_prev) * FTnoappasy_prev(tau)

            FN_source_app = lambda tau: scenario.sCapp * P_prev * FTapp_prev(tau) + (
                    1 - P_prev) * scenario.sCnoapp * FTnoapp_prev(tau)  # CDF of the time of notification for the
            # source, given that the recipient has the app
            FAappsy_t = evolution(FN_source=FN_source_app, FAs_recipient=scenario.FAsapp, time_step=tildetauC_exp)
            FAappasy_t = evolution(FN_source=FN_source_app, FAs_recipient=lambda tau: 0, time_step=tildetauC_exp)

            FN_source_noapp = lambda tau: scenario.sCnoapp * P_prev * FTapp_prev(tau) + (
                    1 - P_prev) * scenario.sCnoapp * FTnoapp_prev(tau)
            FAnoappsy_t = evolution(FN_source=FN_source_noapp, FAs_recipient=scenario.FAsnoapp, time_step=tildetauC_exp)
            FAnoappasy_t = evolution(FN_source=FN_source_noapp, FAs_recipient=lambda tau: 0, time_step=tildetauC_exp)

        FTappsy_t = convolve(FAappsy_t, scenario.p_DeltaATapp, RealRange(0, tau_max, integration_step))
        rappsy_t = suppressed_r_from_test_cdf(lambda tau: r0sy(tau), FTappsy_t, scenario.xi)

        FTappasy_t = convolve(FAappasy_t, scenario.p_DeltaATapp, RealRange(0, tau_max, integration_step))
        rappasy_t = suppressed_r_from_test_cdf(lambda tau: r0asy(tau), FTappasy_t, scenario.xi)

        FTnoappsy_t = convolve(FAnoappsy_t, scenario.p_DeltaATapp, RealRange(0, tau_max, integration_step))
        rnoappsy_t = suppressed_r_from_test_cdf(lambda tau: r0sy(tau), FTnoappsy_t, scenario.xi)

        FTnoappasy_t = convolve(FAnoappasy_t, scenario.p_DeltaATnoapp, RealRange(0, tau_max, integration_step))
        rnoappasy_t = suppressed_r_from_test_cdf(lambda tau: r0asy(tau), FTnoappasy_t, scenario.xi)

        rapp_t = lambda tau: alpha * rappsy_t(tau) + (1 - alpha) * rappasy_t(tau)
        rnoapp_t = lambda tau: alpha * rnoappsy_t(tau) + (1 - alpha) * rnoappasy_t(tau)

        rsy_t = lambda tau: scenario.epsilon(t) * rappsy_t(tau) + (1 - scenario.epsilon(t)) * rnoappsy_t(tau)
        rasy_t = lambda tau: scenario.epsilon(t) * rappasy_t(tau) + (1 - scenario.epsilon(t)) * rnoappasy_t(tau)

        r_t = lambda tau: scenario.epsilon(t) * rapp_t(tau) + (1 - scenario.epsilon(t)) * rnoapp_t(tau)

        Rapp_t = integrate(rapp_t)
        Rnoapp_t = integrate(rnoapp_t)
        Rsy_t = integrate(rsy_t)
        Rasy_t = integrate(rasy_t)
        R_t = integrate(r_t)

        tauC_exp = integrate(lambda tau: tau * r_t(tau) / R_t)

        P_t = scenario.epsilon(t) * Rapp_t / R_t
        Psy_t = alpha * Rsy_t / R_t

        tilder_t = lambda tau: P_t * rapp_t(tau) + (1 - P_t) * rnoapp_t(tau)
        tildeR_t = integrate(tilder_t)
        tildepC_t = lambda tau: tilder_t(tau) / tildeR_t

        FTinftyapp = alpha * FTappsy_t(tau_max) + (1 - alpha) * FTappasy_t(tau_max)
        FTinftynoapp = alpha * FTnoappsy_t(tau_max) + (1 - alpha) * FTnoappasy_t(tau_max)
        FTinfty = scenario.epsilon(t) * FTinftyapp + (1 - scenario.epsilon(t)) * FTinftynoapp

        t_list.append(t)
        r_t_values = list_from_f(r_t, RealRange(0, tau_max, integration_step))
        tildepC_t_values = list_from_f(tildepC_t, RealRange(0, tau_max, integration_step))

        FTappsy_t_values = list_from_f(FTappsy_t, RealRange(0, tau_max, integration_step))
        FTappasy_t_values = list_from_f(FTappasy_t, RealRange(0, tau_max, integration_step))
        FTnoappsy_t_values = list_from_f(FTnoappsy_t, RealRange(0, tau_max, integration_step))
        FTnoappasy_t_values = list_from_f(FTnoappasy_t, RealRange(0, tau_max, integration_step))
        tildepC_list.append(tildepC_t_values)
        FTappsy_list.append(FTappsy_t_values)
        FTappasy_list.append(FTappasy_t_values)
        FTnoappsy_list.append(FTnoappsy_t_values)
        FTnoappasy_list.append(FTnoappasy_t_values)
        FTinftyapp_list.append(FTinftyapp)
        FTinftynoapp_list.append(FTinftynoapp)
        FTinfty_list.append(FTinfty)
        R_list.append(R_t)
        Rapp_list.append(Rapp_t)
        Rnoapp_list.append(Rnoapp_t)
        P_list.append(P_t)
        Psy_list.append(Psy_t)
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
