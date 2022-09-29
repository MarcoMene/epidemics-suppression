# Number of time instants in a day used in the discretization of the model. Increasing this
# number improves the precision:
UNITS_IN_ONE_DAY = 1

TAU_UNIT_IN_DAYS = 1 / UNITS_IN_ONE_DAY

# Max of the support of discrete distributions on non-negative times, when created approximating
# continuous distributions.
TAU_MAX_IN_DAYS = 25
TAU_MAX_IN_UNITS = TAU_MAX_IN_DAYS * UNITS_IN_ONE_DAY


DISTRIBUTION_NORMALIZATION_TOLERANCE = 0.001
FLOAT_TOLERANCE_FOR_EQUALITIES = 1e-10
