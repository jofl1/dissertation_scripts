from pathlib import Path

import numpy as np

import iDEA
import iDEA.interactions
import iDEA.methods.interacting
import iDEA.methods.non_interacting
import iDEA.observables
import iDEA.reverse_engineering
import iDEA.system

# Build the symmetric soft-Coulomb double-well baseline used throughout
# section 3. Solves the interacting two-electron ground state, reverse-
# engineers the Kohn-Sham potential, then symmetrises v_xc and n. The
# inversion plus one-sided gauge correction leaves residual ~1e-3
# asymmetry in v_xc; forcing symmetry makes v_toy(x) - v_toy(-x) = S
# exact, which the toy diagnostic relies on. Expensive (~minutes) -- the
# resulting npz is cached in data/ and reused by every other script in
# this folder.

here = Path(__file__).parent
data_dir = here / "data"
data_path = data_dir / "baseline_symmetric_uu_d10.npz"

# System parameters: L=30 box, two soft-Coulomb wells of depth A=2.0 and
# softening s=1.0, centred at +/-d/2 = +/-5. Grid step dx=0.25 gives 121
# points. Two like-spin (uu) electrons with the standard 1/(|x-x'|+1)
# softened Coulomb interaction.
length = 30.0
well_separation = 10.0
well_depth = 2.0
well_softening = 1.0
dx_target = 0.25
interaction_strength = 1.0
interaction_softening = 1.0


def build_system():
    n_intervals = max(2, int(round(length / dx_target)))
    x = np.linspace(-0.5 * length, 0.5 * length, n_intervals + 1)
    half_d = 0.5 * well_separation
    v_ext = (
        -well_depth / (np.abs(x + half_d) + well_softening)
        - well_depth / (np.abs(x - half_d) + well_softening)
    )
    v_int = iDEA.interactions.softened_interaction(
        x, strength=interaction_strength, softening=interaction_softening,
    )
    return iDEA.system.System(x=x, v_ext=v_ext, v_int=v_int, electrons="uu")


def solve_from_scratch():
    system = build_system()
    x = system.x
    print(f"Grid: {len(x)} points, dx={system.dx:.4f}, x in [{x[0]:.1f}, {x[-1]:.1f}]")

    print("Solving interacting ground state...")
    state = iDEA.methods.interacting.solve(system, k=0)
    n = iDEA.observables.density(system, state=state)
    print(f"  Energy = {state.energy:.6f} Ha")

    print("Reverse-engineering KS potential...")
    s_ks = iDEA.reverse_engineering.reverse(
        system, n, iDEA.methods.non_interacting,
        mu=0.8, pe=0.1, tol=1e-5, silent=False,
    )
    v_ks = s_ks.v_ext
    v_h = iDEA.observables.hartree_potential(system, n)
    v_xc_raw = v_ks - system.v_ext - v_h
    # Pin v_xc to zero at the right boundary -- the gauge convention used
    # everywhere else in the dissertation.
    v_xc = v_xc_raw - v_xc_raw[-1]

    state_ks = iDEA.methods.non_interacting.solve(s_ks, k=0)
    n_ks = iDEA.observables.density(s_ks, state=state_ks)
    ks_err = np.sum(np.abs(n_ks - n)) * system.dx
    print(f"  KS density error: {ks_err:.2e}")

    return {
        "x": x,
        "n": n,
        "v_ext": system.v_ext,
        "v_xc": v_xc,
        "v_xc_raw": v_xc_raw,
        "v_ks": v_ks,
        "v_H": v_h,
        "energy": np.array(state.energy),
        "ks_density_error": np.array(ks_err),
    }


def symmetrise(data):
    # Force exact symmetry. The reverse-engineered v_xc inherits a residual
    # asymmetry from the inversion noise; symmetrising makes the toy step
    # v_toy(x) - v_toy(-x) = S hold to floating-point precision. The density
    # is already symmetric to ~1e-8 but is symmetrised too for parity.
    out = dict(data)
    out["v_xc"] = 0.5 * (data["v_xc"] + data["v_xc"][::-1])
    out["n"] = 0.5 * (data["n"] + data["n"][::-1])
    return out


def main():
    data_dir.mkdir(parents=True, exist_ok=True)

    if data_path.exists():
        print(f"Local baseline already present: {data_path}")
        return

    print("No cached baseline found. Solving via iDEA...")
    data = solve_from_scratch()
    data = symmetrise(data)
    np.savez(data_path, **data)
    print(f"Saved: {data_path}")


if __name__ == "__main__":
    main()
