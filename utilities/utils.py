import numpy as np
from dataclasses import dataclass


@dataclass
class DeltaMeasure:
    """
    Dirac delta positioned in position, with integral height
    """
    position: float
    height: float


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
        return np.arange(self.x_min, self.x_max + self.step, self.step)


def convolve(f1: callable, f2: callable, real_range: RealRange):
    if isinstance(f2, DeltaMeasure):
        return lambda x: f2.height * f1(x - f2.position)
    raise ValueError("Not implemented yet")  # TODO: add


def list_from_f(f: callable, real_range: RealRange) -> list:
    """
    Function to list in a range.
    """
    return [f(x) for x in real_range.x_values]


def f_from_list(f_values: list, real_range: RealRange) -> callable:
    """
    List in a range to function
    """

    def f(x):
        if x < real_range.x_min:
            return f_values[0]
        if x > real_range.x_max:
            return f_values[-1]
        i = int((x - real_range.x_min) / real_range.step)
        return f_values[i]

    return f


def round2(number):
    return round(number, 2)
