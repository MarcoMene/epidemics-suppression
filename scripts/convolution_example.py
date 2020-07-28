from bsp_epidemic_suppression_model.math_utilities.distributions import gamma_pdf
from bsp_epidemic_suppression_model.math_utilities.plotting_utils import plot_functions
from bsp_epidemic_suppression_model.math_utilities.functions_utils import (
    DeltaMeasure,
    convolve,
    RealRange,
)

f = lambda x: gamma_pdf(x - 10, 4, 1)
delta = DeltaMeasure(position=5, height=1)

g = convolve(f, delta)
plot_functions([f, g], RealRange(0, 30, 0.1))
