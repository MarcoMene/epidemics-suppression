from bsp_epidemic_suppression_model.utilities.model import (
    make_scenario_parameters_for_asymptomatics_symptomatics_model,
    effectiveness_from_R,
)
from bsp_epidemic_suppression_model.utilities.scenario import Scenario
from bsp_epidemic_suppression_model.utilities.functions_utils import (
    DeltaMeasure,
    RealRange,
)
from bsp_epidemic_suppression_model.algorithm.time_evolution_with_severity import (
    compute_time_evolution_with_severity,
)
from bsp_epidemic_suppression_model.utilities.functions_utils import round2

import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # gs = [asymptomatic, symptomatic]
    tau_max = 30
    integration_step = 0.1

    p_gs, r0_gs = make_scenario_parameters_for_asymptomatics_symptomatics_model()

    sSs = [0.2, 0.5, 0.8]
    sCs = [0.5, 0.8]
    xis = [0.7, 0.9]
    papps = [0.2, 0.5, 0.7, 0.9]

    for sS in sSs:
        for sC in sCs:
            for xi in xis:
                for papp in papps:

                    scenario = Scenario(
                        p_gs=p_gs,
                        r0_gs=r0_gs,
                        t_0=0,
                        ssapp=[0, sS],
                        ssnoapp=[0, 0.2],
                        scapp=sC,
                        scnoapp=0.2,
                        xi=xi,
                        papp=lambda tau: papp,
                        p_DeltaATapp=DeltaMeasure(position=2),
                        p_DeltaATnoapp=DeltaMeasure(position=4),
                    )

                    step_data_list = compute_time_evolution_with_severity(
                        scenario=scenario,
                        real_range=RealRange(0, tau_max, integration_step),
                        n_iterations=8,
                        verbose=False,
                    )

                    R_last = step_data_list[-1].R
                    eff = effectiveness_from_R(R_last)

                    print(
                        f" {sS} & {sC} & {xi} & {papp} & {round2(R_last)} & {round2(eff)} \\\ "
                    )
