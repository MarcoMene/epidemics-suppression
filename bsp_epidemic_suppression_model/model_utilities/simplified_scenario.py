from dataclasses import dataclass


@dataclass
class SimplifiedScenario:
    R0: float
    ssapp: float
    ssnoapp: float
    scapp: float
    tsapp: float
    tsnoapp: float
    papp: float
    xi: float
