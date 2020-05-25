from dataclasses import dataclass
from bsp_epidemic_suppression_model.utilities import round2
from bsp_epidemic_suppression_model.utilities import gamma_cdf
from bsp_epidemic_suppression_model.utilities import r0_alpha, r0_beta


def approximated_suppressed_R(
    R0: float, FTs_infty: float, FTc_infty: float, xi: float, ts: float
):
    fraction_of_R0_before_isolation_for_symptoms = gamma_cdf(
        ts, alpha=r0_alpha, beta=r0_beta
    )
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


@dataclass
class SimplifiedScenario:
    R0: float
    ssapp: float
    ssnoapp: float
    scapp: float
    tsapp: float
    tsnoapp: float
    papp: float
    xi: float


def simplified_time_evolution(
    simplified_scenario: SimplifiedScenario, n_iterations: int
):
    # Quantities that are constant in time:
    FTsapp_infty = simplified_scenario.ssapp
    FTsnoapp_infty = simplified_scenario.ssnoapp
    FTcnoapp_infty = 0
    FTnoapp_infty = FTsnoapp_infty + FTcnoapp_infty - FTsnoapp_infty * FTcnoapp_infty

    Rnoapp = approximated_suppressed_R(
        R0=simplified_scenario.R0,
        FTs_infty=FTsnoapp_infty,
        FTc_infty=FTcnoapp_infty,
        xi=simplified_scenario.xi,
        ts=simplified_scenario.tsnoapp,
    )

    for i in range(n_iterations):

        if i == 0:
            FTcapp_ti_infty = 0

        else:
            # Previous step data:
            FTapp_tim1_infty = FTapp_ti_infty
            tildepapp_tim1 = tildepapp_ti

            FTcapp_ti_infty = approximated_FTcapp(
                scapp=simplified_scenario.scapp,
                FTapp_tim1_infty=FTapp_tim1_infty,
                FTnoapp_tim1_infty=FTnoapp_infty,
                tildepapp_tim1=tildepapp_tim1,
            )

        # TODO: Why this?? ##################
        if i == 0:
            tildepapp_ti = simplified_scenario.papp
        else:
            tildepapp_ti = simplified_scenario.papp * Rapp_ti / R_ti
        # TODO: #####################

        FTapp_ti_infty = FTsapp_infty + FTcapp_ti_infty - FTsapp_infty * FTcapp_ti_infty

        Rapp_ti = approximated_suppressed_R(
            R0=simplified_scenario.R0,
            FTs_infty=FTsapp_infty,
            FTc_infty=FTcapp_ti_infty,
            xi=simplified_scenario.xi,
            ts=simplified_scenario.tsapp,
        )
        R_ti = (
            simplified_scenario.papp * Rapp_ti + (1 - simplified_scenario.papp) * Rnoapp
        )

        # TODO: I think this is the correct one:
        # tildepapp_ti = simplified_scenario.papp * Rapp_ti / R_ti

        recap = (
            f"i={i}\n"
            f" FTsapp_ti_infty = {round2(FTsapp_infty)}\n"
            f" FTcapp_ti_infty = {round2(FTcapp_ti_infty)}\n"
            f" FTapp_ti_infty = {round2(FTapp_ti_infty)}\n"
            f" Rapp_ti={round2(Rapp_ti)}\n"
            f" Rnoapp_ti={round2(Rnoapp)}\n"
            f" R_ti={round2(R_ti)}\n"
            f" tildepapp_ti={round2(tildepapp_ti)}\n"
        )

        print(recap)


if __name__ == "__main__":
    simplified_scenario = SimplifiedScenario(
        R0=1,
        ssapp=0.7,
        ssnoapp=0.2,
        scapp=0.8,
        tsapp=6.5,
        tsnoapp=8.5,
        papp=0.6,
        xi=0.9,
    )
    simplified_time_evolution(simplified_scenario=simplified_scenario, n_iterations=5)
