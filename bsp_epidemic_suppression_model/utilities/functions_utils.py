from dataclasses import dataclass

from typing import List, Union, Callable

from scipy import integrate as sci_integrate


@dataclass
class RealRange:
    """
    Range in real numbers
    """

    x_min: float
    x_max: float
    step: float

    @property
    def x_values(self):
        return [
            self.x_min + n * self.step
            for n in range(int((self.x_max - self.x_min) / self.step) + 1)
        ]


def round2(number: float) -> float:
    """
    Rounds a number to the second decimal.
    """
    return round(number, 2)


def round2_list(l: List[float]) -> List[float]:
    """
    Rounds a list of numbers to the second decimal.
    """
    return [round2(number) for number in l]


def integrate(f: Callable[[float], float], a: float, b: float) -> float:
    """
    Integral of f from a to b.
    """
    return sci_integrate.quad(f, a, b)[0]


@dataclass
class DeltaMeasure:
    """
    Dirac delta centered in position and rescaled by height.
    """

    position: float
    height: float = 1


# Data types for (improper) PDFs and CDFs
ImproperProbabilityDensity = Union[Callable[[float], float], DeltaMeasure]
ProbabilityCumulativeFunction = Callable[[float], float]
ImproperProbabilityCumulativeFunction = Callable[[float], float]


def convolve(f: Callable[[float], float], delta: DeltaMeasure):
    """
    Computes the convolution of a function f and a DeltaMeasure delta.
    """
    return lambda x: delta.height * f(x - delta.position)


def list_from_f(f: Callable[[float], float], real_range: RealRange) -> List[float]:
    """
    Samples a function over a given range into a list of values.
    """
    return [f(x) for x in real_range.x_values]


def f_from_list(f_values: List[float], real_range: RealRange) -> callable:
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
