from dataclasses import dataclass
from typing import Sequence

from bsp_epidemic_suppression_model.math_utilities.config import (
    FLOAT_TOLERANCE_FOR_EQUALITIES,
)


def normalize_sequence(seq: Sequence[float]) -> Sequence[float]:
    s = sum(seq)
    return type(seq)(x / s for x in seq)


def round2(number: float) -> float:
    """
    Rounds a number to the second decimal.
    """
    return round(number, 2)


def floats_match(x: float, y: float, precision: float = FLOAT_TOLERANCE_FOR_EQUALITIES):
    return abs(x - y) <= precision


def float_sequences_match(
    seq_1: Sequence[float],
    seq_2: Sequence[float],
    precision: float = FLOAT_TOLERANCE_FOR_EQUALITIES,
) -> bool:
    try:
        assert len(seq_1) == len(seq_2)
        for x, y in zip(seq_1, seq_2):
            assert abs(x - y) <= precision
        return True
    except AssertionError:
        return False


@dataclass
class RealRange:
    """
    Range in real numbers.
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
