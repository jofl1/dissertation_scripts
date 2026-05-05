import numpy as np

import iDEA.methods.non_interacting

from utils.system import build_asymmetric_system

# Kinetic operator for the eigenvalue table. Builds an asymmetric system on
# the same grid as the cached npz, then asks iDEA for its kinetic operator
# so the eigenproblem uses the same 13-point stencil as everything else in
# section4. The per-system v_ext does not matter -- only the grid.


def kinetic_operator_from_grid(x):
    s = build_asymmetric_system(1.0, 2.0, 10.0)
    assert np.allclose(s.x, x), "grid mismatch -- eigenvalue table requires the section4 grid"
    return iDEA.methods.non_interacting.kinetic_energy_operator(s)
