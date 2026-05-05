import csv
import re
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Robustness check on the splitting metric's neighbourhood size. The
# master plot uses k=5; this script sweeps k in {3, 5, 7, 10, 15, 20}
# and recomputes the Spearman/Pearson correlation between <sigma_k> and
# the D_1 LOO-CV MAE for each. The headline number is the published
# rho(k=5) -- the rest is to show that the conclusion is not contingent
# on the chosen neighbourhood size.

here = Path(__file__).parent
data_dir = here.parent / "section4" / "data"
results_dir = here / "results"
results_dir.mkdir(parents=True, exist_ok=True)

a_r = 2.0
k_values = [3, 5, 7, 10, 15, 20]
k_published = 5
alpha_grid = np.logspace(-8, -1, 8)
gamma_grid = np.logspace(-3, 2, 12)

filename_pattern = re.compile(
    r"exact_asymmetric_uu_AL([\d.]+)_AR2\.0_d([\d.]+)\.npz$"
)


def discover_systems():
    systems = []
    for f in sorted(data_dir.glob("exact_asymmetric_uu_AL*_AR2.0_d*.npz")):
        if "_L" in f.name:
            continue
        m = filename_pattern.match(f.name)
        if m:
            systems.append((float(m.group(1)), float(m.group(2))))
    return sorted(systems, key=lambda s: (s[0], s[1]))


def compute_splitting_metric(n_in, v_in, k):
    # Average std of v_xc across each point's k-NN neighbourhood in
    # density space. NearestNeighbors uses k+1 so the query point itself
    # is included in its own neighbourhood -- matches fig_5_5.
    X = n_in.reshape(-1, 1)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    _, indices = nn.kneighbors(X)
    stds = np.array([np.std(v_in[indices[i]]) for i in range(len(X))])
    return float(np.mean(stds))


def krr_loo_mae(X, y):
    X_scaled = StandardScaler().fit_transform(X)
    best = np.inf
    for alpha in alpha_grid:
        for gamma in gamma_grid:
            krr = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
            y_pred = cross_val_predict(krr, X_scaled, y, cv=len(y))
            mae = float(np.mean(np.abs(y - y_pred)))
            if mae < best:
                best = mae
    return best


def main():
    systems = discover_systems()
    print(f"Discovered {len(systems)} systems")
    print(f"Sweeping k in {k_values} (published value: k={k_published})")
    print(f"KRR setup matches scripts/section4/fig_5_5_master_plot_expanded.py\n")

    rows = []
    for sys_idx, (a_left, d) in enumerate(systems, start=1):
        path = data_dir / f"exact_asymmetric_uu_AL{a_left}_AR{a_r}_d{d}.npz"
        with np.load(path) as data:
            x = data["x"].copy()
            n = data["n"].copy()
            v_xc_raw = data["v_xc"].copy()
        v_xc = v_xc_raw - v_xc_raw[-1]

        margin = max(3, len(x) // 20)
        sl = slice(margin, -margin)
        x_in, n_in, v_in = x[sl], n[sl], v_xc[sl]

        # The D_1 MAE is independent of k -- compute once per system.
        m1 = krr_loo_mae(n_in.reshape(-1, 1), v_in)

        sigma_per_k = {k: compute_splitting_metric(n_in, v_in, k) for k in k_values}

        sigma_summary = " ".join(f"k{k}={sigma_per_k[k]:.4f}" for k in k_values)
        print(f"[{sys_idx:2d}/{len(systems)}] AL={a_left:.1f} d={d:.1f} "
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

    print(f"\n{'=' * 72}")
    print("ROBUSTNESS: rho(k) between <sigma_k> and D_1 LOO-CV MAE")
    print(f"{'=' * 72}")
    print(f"\n{'k':>4} {'Spearman':>12} {'p_S':>10} "
          f"{'Pearson':>10} {'p_P':>10} {'mean sigma':>12}")
    print("-" * 64)

    summary = []
    for k in k_values:
        sigmas = np.array([r["sigma_k"] for r in rows if r["k"] == k])
        maes = np.array([r["mae_d1"] for r in rows if r["k"] == k])
        rho_s, p_s = spearmanr(sigmas, maes)
        rho_p, p_p = pearsonr(sigmas, maes)
        marker = " <-- published" if k == k_published else ""
        print(f"{k:>4} {rho_s:>12.4f} {p_s:>10.2e} "
              f"{rho_p:>10.4f} {p_p:>10.2e} "
              f"{sigmas.mean():>12.4f}{marker}")
        summary.append({
            "k": k,
            "spearman": float(rho_s),
            "spearman_p": float(p_s),
            "pearson": float(rho_p),
            "pearson_p": float(p_p),
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

    spearmans = [s["spearman"] for s in summary]
    print(f"\n{'=' * 72}")
    print("HEADLINE FOR DISSERTATION")
    print(f"{'=' * 72}")
    print(f"Min Spearman rho across k in {k_values}: {min(spearmans):.3f}")
    print(f"Max Spearman rho across k in {k_values}: {max(spearmans):.3f}")
    published_rho = next(s["spearman"] for s in summary if s["k"] == k_published)
    print(f"Published (k={k_published}) Spearman rho: {published_rho:.3f}")


if __name__ == "__main__":
    main()
    print("Done.")
