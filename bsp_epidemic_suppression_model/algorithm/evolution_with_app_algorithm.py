from typing import List, Optional, Tuple

from bsp_epidemic_suppression_model.algorithm.model_blocks.nu_and_tausigma import (
    compute_tausigma_and_nu_components_at_time_t,
    compute_tausigma_and_nu_components_at_time_t_with_app,
)
from bsp_epidemic_suppression_model.algorithm.model_blocks.testing_time_and_b_t_suppression import (
    compute_suppressed_b_t,
    compute_tauT_t,
)
from bsp_epidemic_suppression_model.algorithm.model_blocks.time_evolution_block import (
    compute_tauAc_t_two_components,
)
from bsp_epidemic_suppression_model.math_utilities.config import UNITS_IN_ONE_DAY
from bsp_epidemic_suppression_model.math_utilities.discrete_distributions_utils import (
    DiscreteDistributionOnNonNegatives,
)
from bsp_epidemic_suppression_model.math_utilities.general_utilities import round2
from bsp_epidemic_suppression_model.model_utilities.scenarios import (
    HomogeneousScenario,
    ScenarioWithApp,
)


def compute_time_evolution_two_component(
    scenario: ScenarioWithApp,
    t_max_in_days: int,
    nu_start: int,
    b_negative_times: Optional[Tuple[DiscreteDistributionOnNonNegatives, ...]] = None,
) -> Tuple[
    List[int],
    List[float],
    List[float],
    List[float],
    List[Tuple[float, ...]],
    List[Tuple[float, ...]],
    List[float],
]:
    """

    :param scenario:
    :param t_max_in_days:
    :param nu_start:
    :param b_negative_times:
    :return: t_in_days_list, nu, nu0, R, R_by_severity, FT_infty
    """
    #
    t_in_days_list = []
    nu = []
    nu_app = []
    nu_noapp = []
    nu0 = []
    b_app: List[Tuple[DiscreteDistributionOnNonNegatives, ...]] = []
    b_noapp: List[Tuple[DiscreteDistributionOnNonNegatives, ...]] = []
    R_by_severity_app: List[Tuple[float, ...]] = []
    R_by_severity_noapp: List[Tuple[float, ...]] = []
    R_app: List[float] = []
    R_noapp: List[float] = []
    R: List[float] = []
    tausigma_app: List[Tuple[DiscreteDistributionOnNonNegatives, ...]] = []
    tausigma_noapp: List[Tuple[DiscreteDistributionOnNonNegatives, ...]] = []
    tauT_app: List[Tuple[DiscreteDistributionOnNonNegatives, ...]] = []
    tauT_noapp: List[Tuple[DiscreteDistributionOnNonNegatives, ...]] = []
    FT_infty: List[float, ...] = []

    gs = range(scenario.n_severities)  # Values of severity G

    t_max = t_max_in_days * UNITS_IN_ONE_DAY
    for t in range(0, t_max + 1):
        t_in_days = t / UNITS_IN_ONE_DAY

        pgs_t_app = tuple(p_g * scenario.papp(t) for p_g in scenario.p_gs)
        pgs_t_noapp = tuple(p_g * (1 - scenario.papp(t)) for p_g in scenario.p_gs)

        # Compute tausigma_t and nu_t from nu_t' and b_t' for t' = 0,...,t-1
        if t == 0 and b_negative_times is None:

            nugsapp_t = tuple(nu_start * p_g for p_g in pgs_t_app)
            nugsnoapp_t = tuple(nu_start * p_g for p_g in pgs_t_noapp)
            nu0_t = nu_start
            tausigmagsapp_t = tausigmagsnoapp_t = DiscreteDistributionOnNonNegatives(
                pmf_values=[], tau_min=0, improper=True
            )
        else:
            (
                nugsapp_t,
                tausigmagsapp_t,
                nugsnoapp_t,
                tausigmagsnoapp_t,
            ) = compute_tausigma_and_nu_components_at_time_t_with_app(
                t=t,
                b_app=b_app,
                b_noapp=b_noapp,
                nu=nu,
                p_gs=scenario.p_gs,
                papp=scenario.papp,
                b_negative_times=b_negative_times,
                nu_negative_times=nu_start,
            )

            nu0_t_gs, _ = compute_tausigma_and_nu_components_at_time_t(
                t=t,
                b=[scenario.b0_gs] * t,
                nu=nu0,
                p_gs=scenario.p_gs,
                b_negative_times=b_negative_times,
                nu_negative_times=nu_start,
            )
            nu0_t = sum(nu0_t_gs)  # People infected at t without isolation measures
        nuapp_t = sum(nugsapp_t)
        nunoapp_t = sum(nugsnoapp_t)

        nu_t = nuapp_t + nunoapp_t  # People infected at t

        if nu_t < 0.5:  # Breaks the loop when nu_t = 0
            break

        # Compute tauAs_t components from tauS
        tauAs_t_gs_app = tuple(
            scenario.tauS.rescale_by_factor(scenario.ssapp[g](t)) for g in gs
        )
        tauAs_t_gs_noapp = tuple(
            scenario.tauS.rescale_by_factor(scenario.ssnoapp[g](t)) for g in gs
        )

        # Time evolution step:
        # Compute tauAc_t from tausigma_t and tauT_t' (for t' = 0,...,t-1) components
        tauAc_t_app, tauAc_t_noapp = compute_tauAc_t_two_components(
            t=t,
            tauT_app=tauT_app,
            tauT_noapp=tauT_noapp,
            tausigmagsapp_t=tausigmagsapp_t,
            tausigmagsnoapp_t=tausigmagsnoapp_t,
            xi=scenario.xi,
            scapp_t=scenario.scapp(t),
            scnoapp_t=scenario.scnoapp(t),
        )

        # Compute tauA_t and tauT_t components from tauAs_t, tauAc_t, and DeltaAT
        tauT_t_gs_app = compute_tauT_t(
            tauAs_t_gs=tauAs_t_gs_app, tauAc_t=tauAc_t_app, DeltaAT=scenario.DeltaATapp
        )
        tauT_t_gs_noapp = compute_tauT_t(
            tauAs_t_gs=tauAs_t_gs_noapp,
            tauAc_t=tauAc_t_noapp,
            DeltaAT=scenario.DeltaATnoapp,
        )

        # Compute b and R
        b_t_gs_app = compute_suppressed_b_t(
            b0_t_gs=scenario.b0_gs, tauT_t_gs=tauT_t_gs_app, xi_t=scenario.xi(t)
        )
        b_t_gs_noapp = compute_suppressed_b_t(
            b0_t_gs=scenario.b0_gs, tauT_t_gs=tauT_t_gs_noapp, xi_t=scenario.xi(t)
        )
        R_t_gs_app = tuple(b_t_g.total_mass for b_t_g in b_t_gs_app)
        R_t_gs_noapp = tuple(b_t_g.total_mass for b_t_g in b_t_gs_noapp)
        R_t_app = sum(p_g * R_t_g for (p_g, R_t_g) in zip(scenario.p_gs, R_t_gs_app))
        R_t_noapp = sum(
            p_g * R_t_g for (p_g, R_t_g) in zip(scenario.p_gs, R_t_gs_noapp)
        )
        R_t = scenario.papp(t) * R_t_app + (1 - scenario.papp(t)) * R_t_noapp
        FT_t_infty = sum(
            p_g
            * (
                scenario.papp(t) * tauT_t_g_app.total_mass
                + (1 - scenario.papp(t)) * tauT_t_g_noapp.total_mass
            )
            for (p_g, tauT_t_g_app, tauT_t_g_noapp) in zip(
                scenario.p_gs, tauT_t_gs_app, tauT_t_gs_noapp
            )
        )

        t_in_days_list.append(t_in_days)
        tausigma_app.append(tausigmagsapp_t)
        tausigma_noapp.append(tausigmagsnoapp_t)
        nu.append(nu_t)
        nu_app.append(nuapp_t)
        nu_noapp.append(nunoapp_t)
        nu0.append(nu0_t)
        b_app.append(b_t_gs_app)
        b_noapp.append(b_t_gs_noapp)
        R_app.append(R_t_app)
        R_noapp.append(R_t_noapp)
        R.append(R_t)
        tauT_app.append(tauT_t_gs_app)
        tauT_noapp.append(tauT_t_gs_noapp)
        FT_infty.append(FT_t_infty)

        if t % UNITS_IN_ONE_DAY == 0:
            EtauC_t_gs_app_in_days = [
                b_t_g.normalize().mean() * UNITS_IN_ONE_DAY for b_t_g in b_t_gs_app
            ]
            EtauC_t_gs_noapp_in_days = [
                b_t_g.normalize().mean() * UNITS_IN_ONE_DAY for b_t_g in b_t_gs_noapp
            ]

            print(
                f"""t = {t_in_days} days
                    nugsapp_t = {tuple(nugsapp_t)},   nugsnoapp_t = {tuple(nugsnoapp_t)},   nu_t = {int(round(nu_t, 0))}
                    nu0_t = {int(round(nu0_t, 0))}
                    R_t_gs_app = {R_t_gs_app},    R_t_app = {R_t_app},    
                    R_t_gs_noapp = {R_t_gs_noapp},    R_t_noapp = {R_t_noapp},        
                    R_t = {round2(R_t)}
                    EtauC_t_gs_app = {tuple(EtauC_t_gs_app_in_days)} days
                    EtauC_t_gs_noapp = {tuple(EtauC_t_gs_noapp_in_days)} days
                    Fsigmagsapp_t(∞) = {tuple(tausigmag_t.total_mass for tausigmag_t in tausigmagsapp_t)}
                    Fsigmagsnoapp_t(∞) = {tuple(tausigmag_t.total_mass for tausigmag_t in tausigmagsnoapp_t)}
                    FAs_t_gs_app(∞) = {tuple(tauAs_t_g_app.total_mass for tauAs_t_g_app in tauAs_t_gs_app)}
                    FAs_t_gs_noapp(∞) = {tuple(tauAs_t_g_noapp.total_mass for tauAs_t_g_noapp in tauAs_t_gs_noapp)}
                    FAc_t_app(∞) = {tauAc_t_app.total_mass},    FAc_t_noapp(∞) = {tauAc_t_noapp.total_mass}
                    FT_t_gs_app(∞) = {tuple(tauT_t_g.total_mass for tauT_t_g in tauT_t_gs_app)},
                    FT_t_gs_noapp(∞) = {tuple(tauT_t_g.total_mass for tauT_t_g in tauT_t_gs_noapp)},
                    FT_t(∞) = {round2(FT_t_infty)}
                    """
            )

    return t_in_days_list, nu, nu0, R, R_by_severity_app, R_by_severity_noapp, FT_infty
