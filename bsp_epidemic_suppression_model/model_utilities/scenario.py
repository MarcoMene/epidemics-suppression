from dataclasses import dataclass
from typing import List, Callable

from bsp_epidemic_suppression_model.math_utilities.functions_utils import (
    ImproperProbabilityDensity,
)


class ScenarioError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__("""The scenario is not well-defined.""", *args)


@dataclass
class Scenario:
    """
    Wraps a simulation scenario, defined by a set of parameters that describe the epidemic and the isolation measures.
    For the meaning of the parameters see the paper: <ref_paper>.
    """

    # Data from the literature:
    p_gs: List[float]  # Probabilities of having given severity
    beta0_gs: List[
        Callable[[float, float], float]
    ]  # Infectiousness distributions (t, tau) -> \beta^0_{t,g}(tau)

    # Model parameters:
    t_0: float  # Absolute time at which isolation policies begin
    ssapp: List[
        float
    ]  # Probabilities of (immediate) CTA after symptoms, given severity and app
    ssnoapp: List[
        float
    ]  # Probabilities of (immediate) CTA after symptoms, given severity and no app
    scapp: float  # Probability of immediate CTA after the source tests positive,
    # given that source and the recipient have the app
    scnoapp: float  # Probability of immediate CTA after the source tests positive,
    # given that one between the source and the recipient does not have the app
    xi: float  # Average reduction of the number of infected people after testing positive

    papp: Callable[
        [float], float
    ]  # papp(t) is the fraction of people with the app at absolute time t

    p_DeltaATapp: ImproperProbabilityDensity
    p_DeltaATnoapp: ImproperProbabilityDensity

    def check(self):
        """
        Checks that all the lists referring to each severity component of the infected population have the same length.
        """
        if (
            not len(self.ssapp)
            == len(self.ssnoapp)
            == len(self.p_gs)
            == len(self.beta0_gs)
        ):
            raise ScenarioError("The lists must have the same length.")

        if sum(self.p_gs) != 1:
            raise ScenarioError(
                "The fractions of people infected with given severity must sum to 1."
            )

    def __post_init__(self):
        self.check()

    @property
    def n_severities(self) -> int:
        """
        Returns the number of segments the infected population is divided into, each depending on the value of the
        severity G.
        """
        return len(self.ssapp)
