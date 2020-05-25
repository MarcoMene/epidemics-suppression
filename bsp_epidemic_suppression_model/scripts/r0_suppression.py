from bsp_epidemic_suppression_model.utilities.model import r0, FS
from bsp_epidemic_suppression_model.algorithm.model_blocks import (
    suppressed_r_from_test_cdf,
)
from bsp_epidemic_suppression_model.utilities.plotting_utils import plot_functions
from bsp_epidemic_suppression_model.utilities.functions_utils import RealRange
import scipy.integrate as integrate
from numpy import heaviside

tau_max = 30
step = 0.05

Deltat_test = 4
ss = 0.2

# tau_s = 10
# F_Tsimple = lambda tau: heaviside(tau - tau_s, 1)

FT = lambda tau: ss * FS(tau - Deltat_test)

xi = 1.  # Probability of (immediate) isolation given positive test

suppressed_r_0 = suppressed_r_from_test_cdf(r0, FT, xi)


suppressed_R_0 = integrate.quad(lambda tau: suppressed_r_0(tau), 0, tau_max)[0]
print("suppressed R_0 =", suppressed_R_0)

plot_functions([r0, suppressed_r_0], RealRange(x_min=0, x_max=tau_max, step=step))

