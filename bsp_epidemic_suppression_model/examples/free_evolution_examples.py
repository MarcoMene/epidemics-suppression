from bsp_epidemic_suppression_model.algorithm.free_evolution_algorithm import (
    free_evolution_by_severity,
    free_evolution_global,
)
from bsp_epidemic_suppression_model.algorithm.model_blocks import check_b_negative_times
from bsp_epidemic_suppression_model.math_utilities.config import UNITS_IN_ONE_DAY
from bsp_epidemic_suppression_model.model_utilities.epidemic_data import (
    b0,
    make_scenario_parameters_for_asymptomatic_symptomatic_model,
)

p_gs, b0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()


def free_evolution_without_severities_example():

    b0_scaled = b0.rescale_by_factor(scale_factor=0.9)

    t_max_in_days = 30
    nu_start = 1000

    free_evolution_global(
        b=[b0_scaled] * t_max_in_days * UNITS_IN_ONE_DAY,
        nu_start=nu_start,
        b_negative_times=b0,
    )


def free_evolution_by_severity_example():

    t_max_in_days = 30
    nu_start = 1000

    b0_gs_scaled = tuple(b0_g.rescale_by_factor(scale_factor=0.9) for b0_g in b0_gs)

    check_b_negative_times(p_gs=p_gs, b_negative_times=b0_gs)

    free_evolution_by_severity(
        b=[b0_gs_scaled] * t_max_in_days * UNITS_IN_ONE_DAY,
        nu_start=nu_start,
        p_gs=p_gs,
        b_negative_times=b0_gs,
    )


if __name__ == "__main__":
    # free_evolution_by_severity_example()

    # free_evolution_by_severity_example()

    pass