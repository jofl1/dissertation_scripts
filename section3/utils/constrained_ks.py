import numpy as np
import scipy.linalg

import iDEA.system
import iDEA.interactions
import iDEA.methods.non_interacting
import iDEA.observables

# Constrained Kohn-Sham SCF loop with a fixed v_xc(x). NOT a self-consistent
# DFT calculation -- v_xc is imposed as a fixed function on the grid rather
# than recomputed as a functional derivative of n. Only the Hartree part
# responds. Uses iDEA primitives (kinetic operator, Hartree potential,
# system construction) but NOT iDEA.methods.non_interacting.solve, so we
# can audit iteration count and residual history.


def make_system(x, v_ext, electrons="uu", strength=1.0, softening=1.0):
    # Softened-Coulomb interaction matching the cached symmetric baseline:
    # 1/(|x-x'|+1), two like-spin electrons.
    v_int = iDEA.interactions.softened_interaction(
        x, strength=strength, softening=softening
    )
    return iDEA.system.System(x, v_ext, v_int, electrons)


def solve_constrained_ks(s, v_xc_fixed, n_init=None, mixing=0.5, tol=1e-10,
                         max_iter=200):
    # H = K + diag(v_ext) + diag(v_H[n]) + diag(v_xc_fixed) at every step;
    # diagonalise with eigh; occupy the lowest s.up_count orbitals (uu so
    # the down sector is empty -- like-spin pair feels the same one-body
    # potential). Linear density mixing. Raises RuntimeError on overrun.
    K = iDEA.methods.non_interacting.kinetic_energy_operator(s)
    v_ext_diag = np.diag(s.v_ext)
    v_xc_diag = np.diag(v_xc_fixed)

    if n_init is None:
        n = np.full_like(s.x, s.count / (s.x[-1] - s.x[0]))
    else:
        n = n_init.copy()

    history = []
    for it in range(max_iter):
        v_h = iDEA.observables.hartree_potential(s, n)
        H = K + v_ext_diag + np.diag(v_h) + v_xc_diag

        _, eigvecs = scipy.linalg.eigh(H)
        # eigh returns L2-orthonormal columns: sum(|psi|^2)*dx = 1. Convert
        # to delta-orthonormal density: n(x) = sum |psi(x)|^2 / dx.
        occ = eigvecs[:, : s.up_count]
        n_new = np.sum(occ * occ, axis=1) / s.dx

        residual = float(np.sum(np.abs(n_new - n)) * s.dx)
        history.append(residual)
        if residual < tol:
            v_ks = s.v_ext + iDEA.observables.hartree_potential(s, n_new) + v_xc_fixed
            return n_new, v_ks, history, it + 1

        n = mixing * n_new + (1.0 - mixing) * n

    raise RuntimeError(
        f"SCF did not converge in {max_iter} iterations; "
        f"final residual = {history[-1]:.3e}"
    )
