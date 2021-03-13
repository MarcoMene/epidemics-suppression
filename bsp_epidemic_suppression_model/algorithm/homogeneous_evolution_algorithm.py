from typing import List, Optional, Tuple

from bsp_epidemic_suppression_model.algorithm.model_blocks.nu_and_tausigma import (
    compute_tausigma_and_nu_components_at_time_t,
)
from bsp_epidemic_suppression_model.algorithm.model_blocks.testing_time_and_b_t_suppression import (
    compute_suppressed_b_t,
    compute_tauT_t,
)
from bsp_epidemic_suppression_model.algorithm.model_blocks.time_evolution_block import (
    compute_tauAc_t,
)
from bsp_epidemic_suppression_model.math_utilities.config import UNITS_IN_ONE_DAY
from bsp_epidemic_suppression_model.math_utilities.discrete_distributions_utils import (
    DiscreteDistributionOnNonNegatives,
)
from bsp_epidemic_suppression_model.math_utilities.general_utilities import round2
from bsp_epidemic_suppression_model.model_utilities.scenarios import HomogeneousScenario


def compute_time_evolution_homogeneous_case(
    scenario: HomogeneousScenario,
    t_max_in_days: int,
    nu_start: int,
    b_negative_times: Optional[Tuple[DiscreteDistributionOnNonNegatives, ...]] = None,
) -> Tuple[
    List[int],
    List[float],
    List[float],
    List[float],
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
    nu0 = []
    b: List[Tuple[DiscreteDistributionOnNonNegatives, ...]] = []
    R_by_severity: List[Tuple[float, ...]] = []
    R: List[float] = []
    tausigma: List[Tuple[DiscreteDistributionOnNonNegatives, ...]] = []
    tauT: List[Tuple[DiscreteDistributionOnNonNegatives, ...]] = []
    FT_infty: List[float, ...] = []

    gs = range(scenario.n_severities)  # Values of severity G

    t_max = t_max_in_days * UNITS_IN_ONE_DAY
    for t in range(0, t_max + 1):
        t_in_days = t / UNITS_IN_ONE_DAY

        # Compute tausigma_t and nu_t from nu_t' and b_t' for t' = 0,...,t-1
        if t == 0 and b_negative_times is None:
            nu_t = nu_start
            nu_t_gs = tuple(nu_t * p_g for p_g in scenario.p_gs)
            nu0_t = nu_start
            tausigmags_t = DiscreteDistributionOnNonNegatives(
                pmf_values=[], tau_min=0, improper=True
            )
        else:
            nu_t_gs, tausigmags_t = compute_tausigma_and_nu_components_at_time_t(
                t=t,
                b=b,
                nu=nu,
                p_gs=scenario.p_gs,
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

            nu_t = sum(nu_t_gs)  # People infected at t
            nu0_t = sum(nu0_t_gs)  # People infected at t without isolation measures

            if nu_t < 0.5:  # Breaks the loop when nu_t = 0
                break

        # Compute tauAs_t components from tauS
        tauAs_t_gs = tuple(
            scenario.tauS.rescale_by_factor(scenario.ss[g](t)) for g in gs
        )

        # Time evolution step:
        # Compute tauAc_t from tausigma_t and tauT_t' (for t' = 0,...,t-1) components
        tauAc_t = compute_tauAc_t(
            t=t,
            tauT=tauT,
            tausigmags_t=tausigmags_t,
            xi=scenario.xi,
            sc_t=scenario.sc(t),
        )

        # Compute tauA_t and tauT_t components from tauAs_t, tauAc_t, and DeltaAT
        tauT_t_gs = compute_tauT_t(
            tauAs_t_gs=tauAs_t_gs, tauAc_t=tauAc_t, DeltaAT=scenario.DeltaAT
        )

        # Compute b and R
        b_t_gs = compute_suppressed_b_t(
            b0_t_gs=scenario.b0_gs, tauT_t_gs=tauT_t_gs, xi_t=scenario.xi(t)
        )
        R_t_gs = tuple(b_t_g.total_mass for b_t_g in b_t_gs)
        R_t = sum(p_g * R_t_g for (p_g, R_t_g) in zip(scenario.p_gs, R_t_gs))
        FT_t_infty = sum(
            p_g * tauT_t_g.total_mass
            for (p_g, tauT_t_g) in zip(scenario.p_gs, tauT_t_gs)
        )

        t_in_days_list.append(t_in_days)
        tausigma.append(tausigmags_t)
        nu.append(nu_t)
        nu0.append(nu0_t)
        b.append(b_t_gs)
        R.append(R_t)
        R_by_severity.append(R_t_gs)
        tauT.append(tauT_t_gs)
        FT_infty.append(FT_t_infty)

        if t % UNITS_IN_ONE_DAY == 0:
            EtauC_t_gs_in_days = [
                b_t_g.normalize().mean() * UNITS_IN_ONE_DAY for b_t_g in b_t_gs
            ]

            print(
                f"""t = {t_in_days} days
                    nu_t_gs = {tuple(nu_t_gs)},   nu_t = {int(round(nu_t, 0))}
                    nu0_t = {int(round(nu0_t, 0))}
                    R_t_gs = {R_t_gs},    R_t = {round2(R_t)}
                    EtauC_t_gs = {tuple(EtauC_t_gs_in_days)} days
                    Fsigmags_t(∞) = {tuple(tausigmag_t.total_mass for tausigmag_t in tausigmags_t)}
                    FAs_t_gs(∞) = {tuple(tauAs_t_g.total_mass for tauAs_t_g in tauAs_t_gs)}
                    FAc_t(∞) = {tauAc_t.total_mass}
                    FT_t_gs(∞) = {tuple(tauT_t_g.total_mass for tauT_t_g in tauT_t_gs)},   FT_t(∞) = {round2(FT_t_infty)}
                    """
            )

    return t_in_days_list, nu, nu0, R, R_by_severity, FT_infty
