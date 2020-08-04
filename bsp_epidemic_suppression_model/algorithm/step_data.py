from typing import List

from bsp_epidemic_suppression_model.math_utilities.functions_utils import (
    RealRange,
    ImproperProbabilityCumulativeFunction,
    list_from_f,
    f_from_list,
)


class StepData:
    """
    Class containing all the data produced at each step of the algorithm.
    Some of these data are functions, but are stored in this class as lists of values.
    """

    def __init__(
        self,
        real_range: RealRange,
        t: float,  # Absolute time of this step
        papp: float,  # Probability that an infected at t has the app
        tildepapp: float,  # Probability that a source infected at t has the app
        tildepgs: List[
            float
        ],  # Probabilities that a source infected at t has each severity
        EtauC: float,  # Expected contagion time for source infected at t
        FT_infty: float,  # Probability that an infected at t tests positive
        FTapp_infty: float,  # Probability that an infected at t with the app tests positive
        FTnoapp_infty: float,  # Probability that an infected at t without the app tests positive
        tildeFTapp: ImproperProbabilityCumulativeFunction,
        # Distribution of testing time for source infected at t with app
        tildeFTnoapp: ImproperProbabilityCumulativeFunction,
        # Distribution of testing time for source infected at t with no app
        R: float,
        Rapp: float,
        Rnoapp: float,
    ):
        self.real_range = real_range
        self.t = t
        self.papp = papp
        self.tildepapp = tildepapp
        self.tildepgs = tildepgs
        self.EtauC = EtauC
        self.FT_infty = FT_infty
        self.FTapp_infty = FTapp_infty
        self.FTnoapp_infty = FTnoapp_infty
        self.tildeFTapp_values = list_from_f(f=tildeFTapp, real_range=real_range)
        self.tildeFTnoapp_values = list_from_f(f=tildeFTnoapp, real_range=real_range)
        self.R = R
        self.Rapp = Rapp
        self.Rnoapp = Rnoapp

    def tildeFTapp(self, tau):
        return f_from_list(f_values=self.tildeFTapp_values, real_range=self.real_range)(
            tau
        )

    def tildeFTnoapp(self, tau):
        return f_from_list(
            f_values=self.tildeFTnoapp_values, real_range=self.real_range
        )(tau)
