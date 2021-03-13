from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Tuple

from bsp_epidemic_suppression_model.math_utilities.discrete_distributions_utils import (
    DiscreteDistributionOnNonNegatives,
)


class ScenarioError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__("""The scenario is not well-defined.""", *args)


@dataclass
class Scenario(ABC):
    """
    Wraps a simulation scenario, defined by a set of parameters that describe the epidemic and
    the isolation measures.
    For the meaning of the parameters see the paper: <ref_paper>.
    """

    # Data from the literature:
    p_gs: Tuple[float, ...]  # Probabilities of having given severity
    b0_gs: Tuple[
        DiscreteDistributionOnNonNegatives, ...
    ]  # Infectiousness distributions (t, tau) -> \beta^0_{t,g}(tau)  # TODO
    tauS: DiscreteDistributionOnNonNegatives

    def __post_init__(self):
        assert sum(self.p_gs) == 1, ScenarioError(
            "The fractions of people infected with given severity must sum to 1."
        )
        self.check_severities()

    def check_severities(self) -> None:
        """
        Checks that all the lists referring to each severity component of the infected population have the same length.
        """
        tuple_length_error = ScenarioError(
            "The tuples depending on severities must have the same length."
        )
        n_severities = self.n_severities
        assert len(self.b0_gs) == n_severities, tuple_length_error
        for t in self.tuples_to_check():
            assert len(t) == n_severities, tuple_length_error

    @abstractmethod
    def tuples_to_check(self) -> Tuple[Tuple, ...]:
        raise NotImplementedError

    @property
    def n_severities(self) -> int:
        """
        Returns the number of segments the infected population is divided into,
        each depending on the value of the severity G.
        """
        return len(self.p_gs)


FunctionOfTimeUnit = Callable[[int], float]


@dataclass
class HomogeneousScenario(Scenario):

    # Model parameters:
    t_0: float  # Absolute time at which isolation policies begin
    ss: Tuple[
        FunctionOfTimeUnit, ...
    ]  # Probabilities of (immediate) CTA after symptoms, given severity
    sc: FunctionOfTimeUnit
    xi: FunctionOfTimeUnit  # Average reduction of the number of infected people after testing positive

    DeltaAT: DiscreteDistributionOnNonNegatives

    def tuples_to_check(self) -> Tuple[Tuple, ...]:
        return (self.ss,)


@dataclass
class ScenarioWithApp(Scenario):

    # Model parameters:
    t_0: float  # Absolute time at which isolation policies begin
    ssapp: Tuple[
        FunctionOfTimeUnit, ...
    ]  # Probabilities of (immediate) CTA after symptoms, given severity and app
    ssnoapp: Tuple[
        FunctionOfTimeUnit, ...
    ]  # Probabilities of (immediate) CTA after symptoms, given severity and no app
    scapp: FunctionOfTimeUnit  # Probability of immediate CTA after the source tests positive,
    # given that source and the recipient have the app
    scnoapp: FunctionOfTimeUnit  # Probability of immediate CTA after the source tests positive,
    # given that one between the source and the recipient does not have the app
    xi: FunctionOfTimeUnit  # Average reduction of the number of infected people after testing positive

    papp: FunctionOfTimeUnit  # papp(t) is the fraction of people with the app at absolute time t

    DeltaATapp: DiscreteDistributionOnNonNegatives
    DeltaATnoapp: DiscreteDistributionOnNonNegatives

    def tuples_to_check(self) -> Tuple[Tuple, ...]:
        return self.ssapp, self.ssnoapp


def homogeneous_to_two_components(
    homogeneous_scenario: HomogeneousScenario,
) -> ScenarioWithApp:
    return ScenarioWithApp(
        p_gs=homogeneous_scenario.p_gs,
        b0_gs=homogeneous_scenario.b0_gs,
        t_0=homogeneous_scenario.t_0,
        ssapp=(lambda t: 0, lambda t: 0),
        ssnoapp=homogeneous_scenario.ss,
        scapp=lambda t: 0,
        scnoapp=homogeneous_scenario.sc,
        xi=homogeneous_scenario.xi,
        papp=lambda tau: 0,
        DeltaATapp=DiscreteDistributionOnNonNegatives(pmf_values=[1], tau_min=0),
        DeltaATnoapp=homogeneous_scenario.DeltaAT,
    )