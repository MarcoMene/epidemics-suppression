from bsp_epidemic_suppression_model.model_utilities.simplified_scenario import (
    SimplifiedScenario,
)
from bsp_epidemic_suppression_model.simplified_algorithm.simplified_algorithm import (
    simplified_time_evolution,
)


def simplified_scenario_example_1():
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


if __name__ == "__main__":

    simplified_scenario_example_1()
