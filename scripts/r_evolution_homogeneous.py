from utilities.model import r0, suppressed_r_from_test_cdf, R0_0, FS
from utilities.plotting_utils import plot_functions, show_plot
from utilities.utils import RealRange, DeltaMeasure, f_from_list, list_from_f, convolve
import scipy.integrate as integrate
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

#   *********************************** CONFIG  ****************************************** #

tau_max = 30
step = 0.05
N_iterations = 6

#   *********************************** INPUTS  ****************************************** #

sS = 0.5  # Probability of (immediate) CTA given symptoms
sC = 0.5
xi = 1.

Deltat_test = 0.

#   *********************************** FUNCTIONS  ****************************************** #

p_DeltaAT = DeltaMeasure(position=Deltat_test, height=1)


def FAs(tau):
    return sS * FS(tau)


#   *********************************** ITERATIONS  ****************************************** #


t_list = []
pC_list = []
FT_list = []
FTinfty_list = []
R_list = []

fig = plt.figure(figsize=(10, 15))

r_tplot = fig.add_subplot(311)
r_tplot.set_xlabel('tau [days]')
r_tplot.set_ylabel('r_t')
r_tplot.grid(True)
r_tplot.set_xlim(0, tau_max)
r_tplot.set_ylim(0, R0_0 / 6)

tau_values = RealRange(0, tau_max, step).x_values

for i in range(0, N_iterations):

    if i == 0:
        t = 0
        FA_t = FAs

    else:
        FT_prev_values = FT_list[-1]
        FT_prev = f_from_list(FT_prev_values, RealRange(0, tau_max, step))
        pC_prev_values = pC_list[-1]
        pC_prev = f_from_list(pC_prev_values, RealRange(0, tau_max, step))

        tauC_exp = integrate.quad(lambda tau: tau * pC_prev(tau), 0, tau_max)[0]
        t = t_list[-1] + tauC_exp
        FAc_t = lambda tau: sC * FT_prev(tau + tauC_exp)  # Time evolution formula here
        FA_t = lambda tau: FAs(tau) + FAc_t(tau) - FAs(tau) * FAc_t(tau)

    FT_t = convolve(FA_t, p_DeltaAT, RealRange(0, tau_max, step))
    FT_t_infty = FT_t(tau_max)
    r_t = suppressed_r_from_test_cdf(lambda tau: r0(tau), FT_t, xi)
    R_t = integrate.quad(r_t, 0, tau_max)[0]
    pC_t = lambda tau: r_t(tau) / R_t

    t_list.append(t)
    r_t_values = list_from_f(r_t, RealRange(0, tau_max, step))
    pC_t_values = list_from_f(pC_t, RealRange(0, tau_max, step))
    FT_t_values = list_from_f(FT_t, RealRange(0, tau_max, step))
    pC_list.append(pC_t_values)
    FT_list.append(FT_t_values)
    FTinfty_list.append(FT_t_infty)
    R_list.append(R_t)

    print(f"t={round(t, 1)}")
    if i > 0:
        print(
            f"    FAs(infty)={FAs(tau_max)}  "
            f"FAc_t(infty)={FAc_t(tau_max)}  "
        )
    print(f"    FA_t(infty)={FA_t(tau_max)} \n"
          f"    FT_t(infty)={round(FT_t_infty, 2)}, \n"
          f"    R_t={round(R_t, 2)}")

    r_tplot.plot(tau_values, r_t_values, "r")

    fig.canvas.draw()

R_tplot = fig.add_subplot(312)
R_tplot.set_xlabel('t [days]')
R_tplot.set_ylabel('R_t')
R_tplot.grid(True)
R_tplot.set_xlim(0, t_list[-1])
R_tplot.set_ylim(0, R0_0)
R_tplot.plot(t_list, R_list, "r")

FTinftyplot = fig.add_subplot(313)
FTinftyplot.set_xlabel('t [days]')
FTinftyplot.set_ylabel('FT_t(infty)')
FTinftyplot.grid(True)
FTinftyplot.set_xlim(0, t_list[-1])
FTinftyplot.set_ylim(0, 1)
FTinftyplot.plot(t_list, FTinfty_list, "r")

show_plot()
