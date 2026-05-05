import csv
from pathlib import Path

import numpy as np

from utils.constants import s_values_response, trim_margin
from utils.descriptors import build_pointwise
from utils.krr import fit_krr_loocv

# KRR (LOO-CV) on the relaxed densities. For each S, fit two models:
# one with the symmetric baseline n^(0)(x) as input (the fixed-density
# case from sections 3.1-3.3), one with the relaxed n^S(x). Target in
# both cases is v_xc^S(x) = v_xc^(0)(x) + (S/2)*sgn(x). Expectation:
# the relaxation moves the density too little for KRR to distinguish
# the two branches, so the relaxed-density D_1 MAE still floors at
# S/2 -- the same lower bound that constrained the fixed-density toy.

here = Path(__file__).parent
baseline_path = here / "data" / "baseline_symmetric_uu_d10.npz"
scf_path = here / "results" / "scf_density_vs_S.npz"
out_path = here / "results" / "relaxed_krr_vs_S.csv"

d1_key = r"$D_1 = [n]$"
oracle_key = r"$D_{\mathrm{oracle}} = [n, x]$"


def main():
    with np.load(baseline_path) as baseline:
        x = baseline["x"]
        n_baseline = baseline["n"]
        v_xc_0 = baseline["v_xc"]
    with np.load(scf_path) as scf:
        n_S_data = {S: scf[f"n_S_{S:.4f}"] for S in s_values_response}

    margin = trim_margin(len(x))
    sl = slice(margin, -margin)
    x_inner = x[sl]
    print(f"Grid: N={len(x)}, inner N={len(x_inner)} (margin={margin})")

    rows = []
    print(f"\n{'S':>6} {'fixed D1':>11} {'relaxed D1':>11} "
          f"{'fixed oracle':>13} {'relaxed oracle':>15} {'S/2':>9}")
    print("-" * 73)

    d_fixed = build_pointwise(x, n_baseline, sl)
    for S in s_values_response:
        n_S = n_S_data[S]
        v_target = v_xc_0 + 0.5 * S * np.sign(x)
        y = v_target[sl]

        res_fix_d1 = fit_krr_loocv(d_fixed[d1_key], y)
        res_fix_or = fit_krr_loocv(d_fixed[oracle_key], y)

        d_relaxed = build_pointwise(x, n_S, sl)
        res_rel_d1 = fit_krr_loocv(d_relaxed[d1_key], y)
        res_rel_or = fit_krr_loocv(d_relaxed[oracle_key], y)

        rows.append({
            "S": float(S),
            "fixed_D1_MAE": res_fix_d1.mae,
            "relaxed_D1_MAE": res_rel_d1.mae,
            "fixed_oracle_MAE": res_fix_or.mae,
            "relaxed_oracle_MAE": res_rel_or.mae,
            "S_over_2": 0.5 * S,
        })

        print(
            f"{S:6.2f} {res_fix_d1.mae:11.4e} {res_rel_d1.mae:11.4e} "
            f"{res_fix_or.mae:13.4e} {res_rel_or.mae:15.4e} {0.5*S:9.4f}"
        )

    cols = ["S", "fixed_D1_MAE", "relaxed_D1_MAE",
            "fixed_oracle_MAE", "relaxed_oracle_MAE", "S_over_2"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in rows:
            w.writerow([row[k] for k in cols])
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
    print("Done.")
