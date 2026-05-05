import numpy as np

import iDEA
import iDEA.methods.interacting
import iDEA.methods.lda
import iDEA.methods.non_interacting
import iDEA.observables
import iDEA.reverse_engineering

from utils.system import build_asymmetric_system

# One-shot solver for an asymmetric double well: interacting ground state,
# KS inversion, LDA reference v_xc, gauge-pin to zero at the right boundary,
# write the npz schema all the figure scripts expect. Skips if the file
# already exists so the four solve_*.py scripts can be re-run cheaply.


def solve_exact(a_left, a_right, d, out_path):
    if out_path.exists():
        print(f"Already exists: {out_path}, skipping.")
        return

    print(f"\n{'=' * 60}")
    print(f"Solving AL={a_left}, AR={a_right}, d={d}")
    print(f"{'=' * 60}")

    s = build_asymmetric_system(a_left, a_right, d)
    print(f"Grid: {len(s.x)} points, dx={s.dx:.4f}")

    print("Solving interacting ground state...")
    state = iDEA.methods.interacting.solve(s, k=0)
    n = iDEA.observables.density(s, state=state)
    print(f"  Energy = {state.energy:.6f} Ha")

    print("Reverse-engineering KS potential...")
    s_ks = iDEA.reverse_engineering.reverse(
        s, n, iDEA.methods.non_interacting,
        mu=0.8, pe=0.1, tol=1e-5, silent=False,
    )
    v_ks = s_ks.v_ext
    v_h = iDEA.observables.hartree_potential(s, n)
    v_xc_raw = v_ks - s.v_ext - v_h
    v_xc = v_xc_raw - v_xc_raw[-1]

    v_xc_lda_raw = iDEA.methods.lda.exchange_correlation_potential(s, n)
    v_xc_lda = v_xc_lda_raw - v_xc_lda_raw[-1]

    state_ks = iDEA.methods.non_interacting.solve(s_ks, k=0)
    n_ks = iDEA.observables.density(s_ks, state=state_ks)
    ks_err = float(np.sum(np.abs(n_ks - n)) * s.dx)
    print(f"  KS density error: {ks_err:.2e}")

    np.savez(
        out_path,
        x=s.x,
        n=n,
        v_ext=s.v_ext,
        v_H=v_h,
        v_ks=v_ks,
        v_xc=v_xc,
        v_xc_raw=v_xc_raw,
        v_xc_lda=v_xc_lda,
        v_xc_lda_raw=v_xc_lda_raw,
        energy=np.array(state.energy),
        ks_density_error=np.array(ks_err),
    )
    print(f"  Saved: {out_path}")
