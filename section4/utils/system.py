import numpy as np

import iDEA
import iDEA.interactions
import iDEA.system

from utils.constants import length, soft, dx_target

# Asymmetric soft-Coulomb double well with two like-spin electrons. Mirrors
# the Case-E construction from jofl1/nearsightedness/system_builders.py but
# inlined here so this folder is self-contained.


def build_asymmetric_system(a_left, a_right, d):
    n_intervals = max(2, int(round(length / dx_target)))
    x = np.linspace(-0.5 * length, 0.5 * length, n_intervals + 1)
    half_d = 0.5 * d
    v_ext = (
        -a_left / (np.abs(x + half_d) + soft)
        - a_right / (np.abs(x - half_d) + soft)
    )
    v_int = iDEA.interactions.softened_interaction(
        x, strength=1.0, softening=1.0,
    )
    return iDEA.system.System(x=x, v_ext=v_ext, v_int=v_int, electrons="uu")
