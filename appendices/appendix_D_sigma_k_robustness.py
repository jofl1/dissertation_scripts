"""KNN splitting-metric sensitivity to neighbourhood size.

Robustness check for Section 4.5 of the dissertation. The published
master plot uses k=5 (six-point neighbourhoods: point + k nearest
neighbours in descriptor space). An examiner could ask whether the
Spearman rank correlation rho=0.94 between <sigma_k> and the local-
model MAE is robust to that choice.

This script recomputes the master-plot quantities for k in
{3, 5, 7, 10, 15, 20} on all 19 exact asymmetric systems, using the
same boundary trim, gauge convention, and KRR setup as
ch5/fig_5_5_master_plot_expanded.py:
    margin = max(3, len(x)//20)
    v_xc gauge: v_xc - v_xc[-1]
    KRR with RBF kernel, alpha grid logspace(-8,-1,8), gamma grid
    logspace(-3,2,12), LOO-CV.

Outputs:
    results/sigma_k_sensitivity.csv          - per-(system, k) <sigma_k>
    results/sigma_k_sensitivity_summary.csv  - per-k Spearman/Pearson
    Printed table of rho(k) for the dissertation appendix.

Self-contained: no imports outside ch5/data/.
"""
from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "ch5" / "data"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AR = 2.0
K_VALUES = [3, 5, 7, 10, 15, 20]  # neighbourhood sizes to sweep
K_PUBLISHED = 5
ALPHA_GRID = np.logspace(-8, -1, 8)
GAMMA_GRID = np.logspace(-3, 2, 12)


def discover_systems():
    pattern = re.compile(
        r"exact_asymmetric_uu_AL([\d.]+)_AR2\.0_d([\d.]+)\.npz$"
    )
    systems = []
    for f in sorted(DATA_DIR.glob("exact_asymmetric_uu_AL*_AR2.0_d*.npz")):
        if "_L" in f.name:
            continue
        m = pattern.match(f.name)
        if m:
            al = float(m.group(1))
            d = float(m.group(2))
            systems.append((al, d))
    return sorted(systems, key=lambda s: (s[0], s[1]))


def load_system(AL, d):
    path = DATA_DIR / f"exact_asymmetric_uu_AL{AL}_AR{AR}_d{d}.npz"
    data = dict(np.load(path))
    v_xc_raw = data["v_xc"]
    data["v_xc"] = v_xc_raw - v_xc_raw[-1]
    return data


def compute_splitting_metric(n_in, v_in, k):
    """<sigma_k> = mean over points of std(v_xc) over (point + k NN).

    Matches the convention of ch5/fig_5_5_master_plot_expanded.py:
    NearestNeighbors is queried with n_neighbors=k+1 so each
    neighbourhood includes the query point itself plus its k nearest
    neighbours in descriptor space.
    """
    X = n_in.reshape(-1, 1)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    _, indices = nn.kneighbors(X)
    stds = np.array([np.std(v_in[indices[i]]) for i in range(len(X))])
    return float(np.mean(stds))


def krr_loo_mae(X, y):
    """Grid-search KRR with LOO-CV; return best MAE."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    best_mae = np.inf
    for alpha in ALPHA_GRID:
        for gamma in GAMMA_GRID:
            krr = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
            y_pred = cross_val_predict(krr, X_scaled, y, cv=len(y))
            mae = np.mean(np.abs(y - y_pred))
            if mae < best_mae:
                best_mae = mae
    return float(best_mae)


def main() -> int:
    systems = discover_systems()
    print(f"Discovered {len(systems)} systems")
    print(f"Sweeping k in {K_VALUES} (published value: k={K_PUBLISHED})")
    print(f"KRR setup matches ch5/fig_5_5_master_plot_expanded.py\n")

    rows = []  # one row per (system, k)
    mae_d1_per_system = {}  # cache: (AL, d) -> D1 MAE

    for sys_idx, (AL, d) in enumerate(systems, start=1):
        data = load_system(AL, d)
        x = data["x"]
        n = data["n"]
        v_xc = data["v_xc"]
        margin = max(3, len(x) // 20)
        sl = slice(margin, -margin)
        x_in, n_in, v_in = x[sl], n[sl], v_xc[sl]

        # MAE only depends on the system, not on k -- compute once.
        m1 = krr_loo_mae(n_in.reshape(-1, 1), v_in)
        mae_d1_per_system[(AL, d)] = m1

        sigma_per_k = {}
        for k in K_VALUES:
            sigma_per_k[k] = compute_splitting_metric(n_in, v_in, k)

        sigma_summary = " ".join(
            f"k{k}={sigma_per_k[k]:.4f}" for k in K_VALUES
        )
        print(f"[{sys_idx:2d}/{len(systems)}] AL={AL:.1f} d={d:.1f} "
              f"MAE_D1={m1:.4f}  {sigma_summary}")

        for k in K_VALUES:
            rows.append({
                "AL": AL, "d": d, "DeltaA": AR - AL,
                "k": k,
                "sigma_k": sigma_per_k[k],
                "mae_d1": m1,
            })

    # ── per-system CSV ──
    csv_path = RESULTS_DIR / "sigma_k_sensitivity.csv"
    cols = ["AL", "d", "DeltaA", "k", "sigma_k", "mae_d1"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")

    # ── per-k summary: Spearman + Pearson rho with D1 MAE ──
    print(f"\n{'='*72}")
    print("ROBUSTNESS: rho(k) between <sigma_k> and D_1 LOO-CV MAE")
    print(f"{'='*72}")
    print(f"\n{'k':>4} {'Spearman':>12} {'p_S':>10} "
          f"{'Pearson':>10} {'p_P':>10} "
          f"{'mean sigma':>12}")
    print("-" * 64)
    summary = []
    for k in K_VALUES:
        sigmas = np.array([r["sigma_k"] for r in rows if r["k"] == k])
        maes = np.array([r["mae_d1"] for r in rows if r["k"] == k])
        rho_s, p_s = spearmanr(sigmas, maes)
        rho_p, p_p = pearsonr(sigmas, maes)
        marker = " <-- published" if k == K_PUBLISHED else ""
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

    summary_path = RESULTS_DIR / "sigma_k_sensitivity_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["k", "spearman", "spearman_p",
                           "pearson", "pearson_p", "mean_sigma"]
        )
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSaved: {summary_path}")

    # ── headline numbers for the appendix ──
    spearmans = [s["spearman"] for s in summary]
    print(f"\n{'='*72}")
    print("HEADLINE FOR DISSERTATION")
    print(f"{'='*72}")
    print(f"Min Spearman rho across k in {K_VALUES}: {min(spearmans):.3f}")
    print(f"Max Spearman rho across k in {K_VALUES}: {max(spearmans):.3f}")
    print(f"Published (k={K_PUBLISHED}) Spearman rho: "
          f"{[s['spearman'] for s in summary if s['k']==K_PUBLISHED][0]:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
