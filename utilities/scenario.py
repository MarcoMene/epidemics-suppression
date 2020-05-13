import numpy as np
from dataclasses import dataclass
from utilities.model import FS
from utilities.utils import DeltaMeasure


@dataclass
class Scenario:
    """
    Set of parameters defining an app distribution scenario
    """
    sSapp: float  # Probability of (immediate) CTA given symptoms
    sSnoapp: float
    sCapp: float
    sCnoapp: float
    xi: float

    epsilon0: float
    t_epsilon: float = 0

    Deltat_testapp: float = 0
    Deltat_testnoapp: float = 4

    def epsilon(self, t):
        if t < self.t_epsilon:
            return 0
        else:
            return self.epsilon0

    @property
    def p_DeltaATapp(self):
        return DeltaMeasure(position=self.Deltat_testapp)

    @property
    def p_DeltaATnoapp(self):
        return DeltaMeasure(position=self.Deltat_testnoapp)

    def FAsapp(self, tau):
        return self.sSapp * FS(tau)

    def FAsnoapp(self, tau):
        return self.sSnoapp * FS(tau)
