import csv
from pathlib import Path

import numpy as np

from utils.constants import s_values_response as s_values
from utils.constrained_ks import make_system, solve_constrained_ks
from utils.density_response import compute_metrics

# Constrained-KS sweep across s_values. The toy step
# v_xc(x) = v_xc^(0)(x) + (S/2)*sgn(x) is imposed as a fixed function and
# the non-interacting density is relaxed self-consistently with Hartree
# feedback. Not a full DFT calculation -- v_xc is fixed, not recomputed
# as a functional derivative.

here = Path(__file__).parent
data_path = here / "data" / "baseline_symmetric_uu_d10.npz"
res_dir = here / "results"
res_dir.mkdir(parents=True, exist_ok=True)


def main():
    with np.load(data_path) as baseline:
        x = baseline["x"]
        n_0 = baseline["n"]
        v_ext = baseline["v_ext"]
        v_xc_0 = baseline["v_xc"]
    dx = float(x[1] - x[0])
    print(f"Grid: N={len(x)}, dx={dx:.4f}, x in [{x[0]:.1f}, {x[-1]:.1f}]")
    print(f"Baseline electron count: {np.sum(n_0)*dx:.6f}")

    s = make_system(x, v_ext, electrons="uu")

    metrics_rows = []
    history_rows = []

    def converge_at(step, n_init):
        # Walk a fallback ladder of (mixing, tol, max_iter). The sharp-step
        # end of the sweep occasionally needs the slower settings.
        v_xc_local = v_xc_0 + 0.5 * step * np.sign(x)
        attempts = [
            {"mixing": 0.5, "tol": 1e-10, "max_iter": 200},
            {"mixing": 0.2, "tol": 1e-9, "max_iter": 500},
            {"mixing": 0.05, "tol": 1e-8, "max_iter": 2000},
        ]
        for i, attempt in enumerate(attempts):
            try:
                out = solve_constrained_ks(s, v_xc_local, n_init=n_init, **attempt)
                print(
                    f"  attempt {i+1}: converged in {out[3]} iters, "
                    f"residual={out[2][-1]:.2e} "
                    f"(mixing={attempt['mixing']}, tol={attempt['tol']})"
                )
                return out
            except RuntimeError as exc:
                print(f"  attempt {i+1} failed: {exc}")
        return None

    # Solve S=0 first to get the SCF reference density. The cached baseline
    # came from reverse-engineering with tol=1e-5, so the constrained-KS
    # fixed point sits ~1e-6 away from it. Using the SCF S=0 result as the
    # reference removes that numerical offset from the response metrics.
    print("\n=== S = 0.0000 (SCF reference run) ===")
    ref = converge_at(0.0, None)
    if ref is None:
        raise RuntimeError("SCF did not converge at S=0; cannot proceed")
    n_0_scf, v_ks_0, history0, n_iters0 = ref
    print(
        f"  n_0_scf vs cached baseline: "
        f"max |Δ| = {float(np.max(np.abs(n_0_scf - n_0))):.2e}"
    )

    densities = {"x": x, "n_baseline": n_0, "n_0_scf": n_0_scf}

    n_warm = n_0_scf
    for S in s_values:
        print(f"\n=== S = {S:.4f} ===")
        if S == 0.0:
            n_S, v_ks_S, history, n_iters = n_0_scf, v_ks_0, history0, n_iters0
        else:
            out = converge_at(S, n_warm)
            if out is None:
                raise RuntimeError(
                    f"SCF did not converge at S={S:.4f}. "
                    f"This sweep stays inside the converged regime; if you "
                    f"are seeing this, the bifurcation has moved below "
                    f"S={S:.4f}."
                )
            n_S, v_ks_S, history, n_iters = out

        densities[f"n_S_{S:.4f}"] = n_S
        densities[f"v_ks_S_{S:.4f}"] = v_ks_S
        n_warm = n_S.copy()

        row = compute_metrics(x, dx, n_0_scf, n_S, S)
        row["iters"] = int(n_iters)
        row["final_residual"] = float(history[-1])
        metrics_rows.append(row)
        for i, r in enumerate(history):
            history_rows.append({"S": S, "iter": i + 1, "residual": r})

        print(
            f"  ‖Δn‖₁={row['dn_L1']:.3e}  ‖Δn‖∞={row['dn_Linf']:.3e}  "
            f"ΔN_L={row['dN_left']:+.3e}"
        )
        print(
            f"  Q={row['Q']:.3e}  S/Q={row['step_per_Q']:.2f} Ha/e  "
            f"δn_mirror^max={row['mirror_dn_max']:.3e}"
        )

    np.savez(res_dir / "scf_density_vs_S.npz", **densities)
    print(f"\nSaved densities to {res_dir/'scf_density_vs_S.npz'}")

    csv_path = res_dir / "response_metrics_vs_S.csv"
    cols = [
        "S",
        "dn_L1",
        "dn_Linf",
        "dN_left",
        "Q",
        "step_per_Q",
        "mirror_dn_max",
        "mirror_dn_mean",
        "step_per_mirror_dn_max",
        "iters",
        "final_residual",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in metrics_rows:
            w.writerow([row[k] for k in cols])
    print(f"Saved metrics to {csv_path}")

    history_path = res_dir / "scf_history.csv"
    with open(history_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["S", "iter", "residual"])
        for row in history_rows:
            w.writerow([row["S"], row["iter"], row["residual"]])
    print(f"Saved SCF history to {history_path}")


if __name__ == "__main__":
    main()
    print("Done.")
