from typing import List, Callable

from bsp_epidemic_suppression_model.utilities.functions_utils import (
    RealRange,
    convolve,
    ImproperProbabilityDensity,
    ProbabilityCumulativeFunction,
    ImproperProbabilityCumulativeFunction,
    integrate,
)


def compute_FA_from_FAs_and_previous_step_data(
    FAsapp_ti_gs: List[ImproperProbabilityCumulativeFunction],
    FAsnoapp_ti_gs: List[ImproperProbabilityCumulativeFunction],
    tildepapp_tim1: float,
    tildeFTapp_tim1: ProbabilityCumulativeFunction,
    tildeFTnoapp_tim1: ProbabilityCumulativeFunction,
    EtauC_tim1: float,
    scapp: float,
    scnoapp: float,
) -> (
    List[ImproperProbabilityCumulativeFunction],
    List[ImproperProbabilityCumulativeFunction],
):
    """
    Implements the formulae ...
    :param FAsapp_ti_gs:
    :param FAsnoapp_ti_gs:
    :param tildepapp_tim1: Probability that the source has the app
    :param tildeFTapp_tim1: CDF of the testing time of the source with app
    :param tildeFTnoapp_tim1: CDF of the testing time of the source with no app
    :param EtauC_tim1:
    :return: FAapp_ti_gs, FAnoapp_ti_gs
    """
    gs = range(len(FAsapp_ti_gs))  # Values of severity G

    # CDF of the time of notification for the source, given that the recipient has the app:
    FNapp_tim1 = lambda tau: (
        scapp * tildepapp_tim1 * tildeFTapp_tim1(tau)
        + scnoapp * (1 - tildepapp_tim1) * tildeFTnoapp_tim1(tau)
    )

    # CDF of the time of notification for the source, given that the recipient does not have the app
    # (this is simply scnoapp * tildeFT_tim1):
    FNnoapp_tim1 = lambda tau: (
        scnoapp
        * (
            tildepapp_tim1 * tildeFTapp_tim1(tau)
            + (1 - tildepapp_tim1) * tildeFTnoapp_tim1(tau)
        )
    )

    # Evolution equations here:
    def evolution(FN_tim1_component: callable, FAs_ti_component, EtauC):
        FAc_ti_component = lambda tau: FN_tim1_component(tau + EtauC)
        FA_ti_component = lambda tau: (
            FAs_ti_component(tau)
            + FAc_ti_component(tau)
            - FAs_ti_component(tau) * FAc_ti_component(tau)
        )
        return FA_ti_component

    FAapp_ti_gs = []
    FAnoapp_ti_gs = []

    for g in gs:
        FAapp_ti_g = evolution(
            FN_tim1_component=FNapp_tim1,
            FAs_ti_component=FAsapp_ti_gs[g],
            EtauC=EtauC_tim1,
        )
        FAapp_ti_gs.append(FAapp_ti_g)
        FAnoapp_ti_g = evolution(
            FN_tim1_component=FNnoapp_tim1,
            FAs_ti_component=FAsnoapp_ti_gs[g],
            EtauC=EtauC_tim1,
        )
        FAnoapp_ti_gs.append(FAnoapp_ti_g)

    return FAapp_ti_gs, FAnoapp_ti_gs


def compute_FT_from_FA_and_DeltaAT(
    FAapp_ti_gs: List[callable],
    FAnoapp_ti_gs: List[callable],
    p_DeltaATapp: ImproperProbabilityDensity,
    p_DeltaATnoapp: ImproperProbabilityDensity,
    real_range: RealRange,
):
    gs = range(len(FAapp_ti_gs))  # Values of severity G

    FTapp_ti_gs = []
    FTnoapp_ti_gs = []
    for g in gs:
        FTapp_ti_g = convolve(
            f1=FAapp_ti_gs[g], f2=p_DeltaATapp, real_range=real_range,
        )
        FTapp_ti_gs.append(FTapp_ti_g)
        FTnoapp_ti_g = convolve(
            f1=FAnoapp_ti_gs[g], f2=p_DeltaATnoapp, real_range=real_range,
        )
        FTnoapp_ti_gs.append(FTnoapp_ti_g)

    return FTapp_ti_gs, FTnoapp_ti_gs


def suppressed_r_from_test_cdf(
    r0_component: callable, FT_component: callable, xi: float
) -> callable:
    """
    Given a starting r0 density and a test CDF, calculates the new r profile, suppressed
    """
    return lambda tau: r0_component(tau) * (1 - FT_component(tau) * xi)


def compute_r_R_components_from_FT(
    FTapp_ti_gs: List[ImproperProbabilityCumulativeFunction],
    FTnoapp_ti_gs: List[ImproperProbabilityCumulativeFunction],
    r0_ti_gs: List[Callable[[float], float]],
    xi: float,
    tau_max: float,
):
    gs = range(len(FTapp_ti_gs))  # Values of severity G

    rapp_ti_gs = []
    rnoapp_ti_gs = []
    Rapp_ti_gs = []
    Rnoapp_ti_gs = []
    for g in gs:
        # rapp_ti_g = lambda tau, g=g: suppressed_r_from_test_cdf(
        #     r0_component=r0_ti_gs[g], FT_component=FTapp_ti_gs[g], xi=xi
        # )(tau)
        rapp_ti_g = suppressed_r_from_test_cdf(
            r0_component=r0_ti_gs[g], FT_component=FTapp_ti_gs[g], xi=xi
        )
        rapp_ti_gs.append(rapp_ti_g)
        rnoapp_ti_g = suppressed_r_from_test_cdf(
            r0_component=r0_ti_gs[g], FT_component=FTnoapp_ti_gs[g], xi=xi
        )
        rnoapp_ti_gs.append(rnoapp_ti_g)

        Rapp_ti_g = integrate(f=rapp_ti_g, a=0, b=tau_max)
        Rapp_ti_gs.append(Rapp_ti_g)
        Rnoapp_ti_g = integrate(f=rnoapp_ti_g, a=0, b=tau_max)
        Rnoapp_ti_gs.append(Rnoapp_ti_g)

    return rapp_ti_gs, rnoapp_ti_gs, Rapp_ti_gs, Rnoapp_ti_gs
