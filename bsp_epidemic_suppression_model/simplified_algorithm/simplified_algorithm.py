from bsp_epidemic_suppression_model.math_utilities.functions_utils import round2

from bsp_epidemic_suppression_model.model_utilities.simplified_scenario import (
    SimplifiedScenario,
)

from bsp_epidemic_suppression_model.simplified_algorithm.approximate_model_blocks import (
    approximated_suppressed_R,
    approximated_FTcapp,
)


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
