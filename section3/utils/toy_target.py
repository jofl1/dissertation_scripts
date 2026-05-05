import numpy as np

# Antisymmetric Heaviside toy-target builder. np.sign(0.0) == 0.0, so the
# central grid point inherits the baseline value with no discontinuity
# sitting on it. For any mirror pair (x, -x) with x != 0, this guarantees
# v_toy(x) - v_toy(-x) == S to floating-point precision.


def make_toy_target(v_xc_baseline, x, S):
    return v_xc_baseline + 0.5 * S * np.sign(x)
