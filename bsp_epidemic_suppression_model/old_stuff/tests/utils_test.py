from bsp_epidemic_suppression_model.math_utilities.general_utilities import RealRange
from bsp_epidemic_suppression_model.old_stuff.functions_utils import (
    f_from_list,
    list_from_f,
)


def test_functions_to_lists():
    def f(x):
        return x * x

    real_range = RealRange(0, 2, 0.5)

    lf = list_from_f(f, real_range)

    assert len(lf) == 5
    assert all([lf[i] == real_range.x_values[i] ** 2 for i in range(len(lf))])

    fl = f_from_list(lf, real_range)

    assert all(
        [
            fl(real_range.x_values[i]) == real_range.x_values[i] ** 2
            for i in range(len(lf))
        ]
    )
