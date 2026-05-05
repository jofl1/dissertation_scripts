from pathlib import Path

import numpy as np
import scipy.linalg

from utils.constants import a_r
from utils.kinetic import kinetic_operator_from_grid

# Single-particle eigenvalue ordering for tab:eigenvalues. For each system
# we diagonalise H = K + diag(v) in v_ext, v_ext + v_H, and v_KS, then
# label the two lowest orbitals by which well holds the bulk of the
# probability. The dissertation table picks values out of stdout by hand.

here = Path(__file__).parent
data_dir = here / "data"

# (a_left, d): DeltaA=0.5 d=10 has no reordering; DeltaA=1.0 sweeps the
# crossover from L,L through partial mixing to full R,R reordering.
systems = [
    (1.5, 10.0),
    (1.0, 6.0),
    (1.0, 8.0),
    (1.0, 10.0),
]


def solve_eigenvalues(x, v_potential):
    K = kinetic_operator_from_grid(x)
    H = K + np.diag(v_potential)
    energies, orbitals = scipy.linalg.eigh(H)
    return energies, orbitals / np.sqrt(x[1] - x[0])


def localisation(x, orbital):
    # 'L' or 'R' if >90% of |orbital|^2 sits on one side of x=0; otherwise
    # report the dominant-side fraction so the caller can flag mixed states.
    dx = x[1] - x[0]
    density = orbital ** 2 * dx
    left_weight = float(np.sum(density[x < 0]))
    right_weight = float(np.sum(density[x > 0]))
    if left_weight > 0.9:
        return "L"
    if right_weight > 0.9:
        return "R"
    return f"L{left_weight:.0%}/R{right_weight:.0%}"


def main():
    print(f"{'System':>16} | {'ε₁(v_ext)':>10} {'ε₂(v_ext)':>10} {'loc':>8} | "
          f"{'ε₁(v_ext+v_H)':>13} {'ε₂(v_ext+v_H)':>13} {'loc':>8} | "
          f"{'reorders':>8} | {'ε₁(v_KS)':>10} {'ε₂(v_KS)':>10} {'loc':>8}")
    print("-" * 145)

    for a_left, d in systems:
        delta = a_r - a_left
        path = data_dir / f"exact_asymmetric_uu_AL{a_left}_AR{a_r}_d{d}.npz"
        with np.load(path) as data:
            x = data["x"].copy()
            v_ext = data["v_ext"].copy()
            v_h = data["v_H"].copy()
            v_ks = data["v_ks"].copy()

        e_ext, orb_ext = solve_eigenvalues(x, v_ext)
        loc_ext_1 = localisation(x, orb_ext[:, 0])
        loc_ext_2 = localisation(x, orb_ext[:, 1])

        e_h, orb_h = solve_eigenvalues(x, v_ext + v_h)
        loc_h_1 = localisation(x, orb_h[:, 0])
        loc_h_2 = localisation(x, orb_h[:, 1])
        reorders = (loc_h_1 == loc_h_2 == "R") or (loc_h_1 == loc_h_2 == "L")

        e_ks, orb_ks = solve_eigenvalues(x, v_ks)
        loc_ks_1 = localisation(x, orb_ks[:, 0])
        loc_ks_2 = localisation(x, orb_ks[:, 1])

        label = f"ΔA={delta:.1f}, d={d:.0f}"
        print(f"{label:>16} | {e_ext[0]:10.4f} {e_ext[1]:10.4f} "
              f"{loc_ext_1 + ',' + loc_ext_2:>8} | "
              f"{e_h[0]:13.4f} {e_h[1]:13.4f} "
              f"{loc_h_1 + ',' + loc_h_2:>8} | "
              f"{'YES' if reorders else 'no':>8} | "
              f"{e_ks[0]:10.4f} {e_ks[1]:10.4f} "
              f"{loc_ks_1 + ',' + loc_ks_2:>8}")
        print(f"{'':>16}   gap: {e_ext[1] - e_ext[0]:10.4f}{'':>20} "
              f"gap: {e_h[1] - e_h[0]:13.4f}")
        print()


if __name__ == "__main__":
    main()
    print("Done.")
