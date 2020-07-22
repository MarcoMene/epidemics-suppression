from bsp_epidemic_suppression_model.math_utilities.distributions import gamma_cdf
from bsp_epidemic_suppression_model.model_utilities.epidemic_data import alpha, beta


def approximated_suppressed_R(
    R0: float, FTs_infty: float, FTc_infty: float, xi: float, ts: float
):
    fraction_of_R0_before_isolation_for_symptoms = gamma_cdf(ts, alpha=alpha, beta=beta)
    symptoms_rescaling = fraction_of_R0_before_isolation_for_symptoms + (
        1 - fraction_of_R0_before_isolation_for_symptoms
    ) * (1 - xi * FTs_infty)
    contacts_rescaling = 1 - xi * FTc_infty

    R = R0 * symptoms_rescaling * contacts_rescaling
    return R


def approximated_FTcapp(
    scapp: float,
    FTapp_tim1_infty: float,
    FTnoapp_tim1_infty: float,
    tildepapp_tim1: float,
):
    FTcapp_ti_infty = (
        tildepapp_tim1 * FTapp_tim1_infty + (1 - tildepapp_tim1) * FTnoapp_tim1_infty
    ) * scapp
    return FTcapp_ti_infty
