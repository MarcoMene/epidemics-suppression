import matplotlib.pyplot as plt

from utilities.utils import RealRange


def plot_functions(fs: list, real_range: RealRange):
    """
    Util to plot a list of functions in a range, with a step.
    """
    for i, f in enumerate(fs):
        plt.plot(real_range.x_values, [f(x) for x in real_range.x_values], label=str(i))
    plt.legend()


def show_plot():
    print("Enjoy the plot!")
    plt.show()
