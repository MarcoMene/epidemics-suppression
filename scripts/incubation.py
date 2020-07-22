from bsp_epidemic_suppression_model.utilities.epidemic_data import FS
from bsp_epidemic_suppression_model.utilities.plotting_utils import plot_functions
from bsp_epidemic_suppression_model.utilities.functions_utils import (
    RealRange,
    integrate,
)

tau_max = 30
step = 0.05


plot_functions(
    [FS],
    RealRange(x_min=0, x_max=tau_max, step=step),
    labels=["FS"],
    title="The CDF of tau^S",
)
EtauS = integrate(
    lambda tau: (1 - FS(tau)), 0, tau_max
)  # Expected time of symptoms onset for symptomatics
print("E(tauS) =", EtauS)
