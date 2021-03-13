from dataclasses import dataclass
from typing import Callable, List, Sequence, Union

from scipy import integrate as sci_integrate

from bsp_epidemic_suppression_model.math_utilities.general_utilities import (
    RealRange,
)


@dataclass
class DeltaMeasure:
    """
    Dirac delta centered in position and rescaled by height.
    """

    position: float
    height: float = 1


# Data types for (improper) PDFs and CDFs
ImproperProbabilityDensity = Union[Callable[[int], float], DeltaMeasure]
ProbabilityCumulativeFunction = Callable[[int], float]
ImproperProbabilityCumulativeFunction = Callable[[int], float]


def list_from_f(f: Callable[[float], float], real_range: RealRange) -> List[float]:
    """
    Samples a function f over a given range into a list of values.
    """
    return [f(x) for x in real_range.x_values]


def f_from_list(
    f_values: List[float], real_range: RealRange
) -> Callable[[float], float]:
    """
    Interpolates a list of samples over a range into a function.
    """

    def f(x):
        if x < real_range.x_min:
            return f_values[0]
        if x > real_range.x_max:
            return f_values[-1]
        i = int((x - real_range.x_min) / real_range.step)
        return f_values[i]

    return f


def round2_sequence(l: Sequence[float]) -> List[float]:
    """
    Rounds a list of numbers to the second decimal.
    """
    return [round(number, 2) for number in l]


def convolve(f: Callable[[float], float], delta: DeltaMeasure):
    """
    Computes the convolution of a function f and a DeltaMeasure delta.
    """
    return lambda x: delta.height * f(x - delta.position)


def integrate(f: Callable[[float], float], a: float, b: float) -> float:
    """
    Integral of a function f from a to b.
    """
    return sci_integrate.quad(f, a, b)[0]
