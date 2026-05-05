import csv
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from utils.constants import a_r, gamma_grid_12
from utils.io import discover_systems, load_inner
from utils.krr import fit_krr_loocv
from utils.metrics import splitting_metric_knn

# Robustness check on the splitting metric's neighbourhood size. The
# master plot uses k=5; this script sweeps k in {3, 5, 7, 10, 15, 20}
# and recomputes the Spearman/Pearson correlation between <sigma_k>
# and the D_1 LOO-CV MAE for each. The headline number is the published
# rho(k=5) -- the rest is to show the conclusion is not contingent on
# the chosen neighbourhood size.

here = Path(__file__).parent
data_dir = here.parent / "section4" / "data"
results_dir = here / "results"
results_dir.mkdir(parents=True, exist_ok=True)

k_values = [3, 5, 7, 10, 15, 20]
k_published = 5


def main():
    systems = sorted(discover_systems(data_dir), key=lambda s: (s[0], s[1]))
    print(f"Discovered {len(systems)} systems")

    rows = []
    for sys_idx, (a_left, d) in enumerate(systems, start=1):
        x_in, n_in, v_in = load_inner(data_dir, a_left, d)

        # D_1 MAE is independent of k -- compute once per system.
        m1 = fit_krr_loocv(n_in.reshape(-1, 1), v_in, gammas=gamma_grid_12).mae
        sigma_per_k = {k: splitting_metric_knn(n_in, v_in, k) for k in k_values}

        sigma_summary = " ".join(f"k{k}={sigma_per_k[k]:.4f}" for k in k_values)
        print(f"[{sys_idx:2d}/{len(systems)}] AL={a_left:.1f} d={d:.1f}  "
              f"MAE_D1={m1:.4f}  {sigma_summary}")

        for k in k_values:
            rows.append({
                "AL": a_left, "d": d, "DeltaA": a_r - a_left,
                "k": k,
                "sigma_k": sigma_per_k[k],
                "mae_d1": m1,
            })

    csv_path = results_dir / "sigma_k_sensitivity.csv"
    cols = ["AL", "d", "DeltaA", "k", "sigma_k", "mae_d1"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in rows:
            w.writerow([row[c] for c in cols])
    print(f"\nSaved: {csv_path}")

    print(f"\n{'k':>4} {'Spearman':>12} {'p_S':>10} "
          f"{'Pearson':>10} {'p_P':>10} {'mean sigma':>12}")
    summary = []
    for k in k_values:
        sigmas = np.array([r["sigma_k"] for r in rows if r["k"] == k])
        maes = np.array([r["mae_d1"] for r in rows if r["k"] == k])
        rho_s, p_s = spearmanr(sigmas, maes)
        rho_p, p_p = pearsonr(sigmas, maes)
        marker = " <-- published" if k == k_published else ""
        print(f"{k:>4} {rho_s:12.4f} {p_s:10.2e} "
              f"{rho_p:10.4f} {p_p:10.2e} "
              f"{sigmas.mean():12.4f}{marker}")
        summary.append({
            "k": k,
            "spearman": float(rho_s), "spearman_p": float(p_s),
            "pearson": float(rho_p), "pearson_p": float(p_p),
            "mean_sigma": float(sigmas.mean()),
        })

    summary_path = results_dir / "sigma_k_sensitivity_summary.csv"
    summary_cols = ["k", "spearman", "spearman_p",
                    "pearson", "pearson_p", "mean_sigma"]
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(summary_cols)
        for row in summary:
            w.writerow([row[c] for c in summary_cols])
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
    print("Done.")
