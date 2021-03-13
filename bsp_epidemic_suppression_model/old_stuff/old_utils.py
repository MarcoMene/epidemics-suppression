from typing import Callable, Sequence

from bsp_epidemic_suppression_model.math_utilities.general_utilities import RealRange

PMF = Callable[[int], float]


def value_from_sequence(
    f_values: Sequence[float], real_range: RealRange, x: float
) -> float:
    """
    Interpolates a list of samples over a range into a function.
    """
    if x < real_range.x_min:
        return f_values[0]
    if x > real_range.x_max:
        return f_values[-1]
    i = int((x - real_range.x_min) / real_range.step)
    return f_values[i]
