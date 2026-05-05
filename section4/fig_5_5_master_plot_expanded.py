import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from utils.constants import (
    a_r, d_default, gamma_grid_12, k_neighbours, rc_params, trim_margin,
)
from utils.krr import fit_krr_loocv
from utils.metrics import splitting_metric_knn

# Master scatter pooling all 19 exact systems: KRR LOO-CV MAE vs splitting
# metric for D_1=[n] and the oracle. Auto-discovers npz files in data/ so
# the figure picks up the asymmetry scan, both distance scans, the cross
# grid, and the make_baseline representative system without an explicit
# parameter list. Marker style records the system's provenance.

here = Path(__file__).parent
data_dir = here / "data"
fig_dir = here / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

asymmetry_scan_d = d_default
distance_scan_als = {1.0, 1.5}

filename_pattern = re.compile(
    r"exact_asymmetric_uu_AL([\d.]+)_AR2\.0_d([\d.]+)\.npz$"
)


def discover_systems():
    systems = []
    for f in sorted(data_dir.glob("exact_asymmetric_uu_AL*_AR2.0_d*.npz")):
        # Skip earlier L=50 box-size test files if they happen to be lying
        # around in data/. The refactored solve scripts never produce them.
        if "_L" in f.name:
            continue
        m = filename_pattern.match(f.name)
        if m:
            systems.append((float(m.group(1)), float(m.group(2))))
    return systems


def classify_system(a_left, d):
    in_asym = (d == asymmetry_scan_d)
    in_dist = (a_left in distance_scan_als)
    if in_asym and in_dist:
        return "both"
    if in_dist:
        return "distance"
    if in_asym:
        return "asymmetry"
    return "cross-grid"


def main():
    plt.rcParams.update(rc_params)

    systems = discover_systems()
    print(f"Discovered {len(systems)} systems:")
    for a_left, d in systems:
        print(f"  AL={a_left}, d={d}, DeltaA={a_r - a_left:.1f}, "
              f"class={classify_system(a_left, d)}")

    results = []
    for a_left, d in systems:
        print(f"\nProcessing AL={a_left}, d={d} ...")
        path = data_dir / f"exact_asymmetric_uu_AL{a_left}_AR{a_r}_d{d}.npz"
        with np.load(path) as data:
            x = data["x"]
            n = data["n"]
            v_xc_raw = data["v_xc"]
        v_xc = v_xc_raw - v_xc_raw[-1]

        sl = slice(trim_margin(len(x)), -trim_margin(len(x)))
        x_in, n_in, v_in = x[sl], n[sl], v_xc[sl]

        sm = splitting_metric_knn(n_in, v_in, k=k_neighbours)
        print(f"  Splitting: {sm:.4f}")

        m1 = fit_krr_loocv(n_in.reshape(-1, 1), v_in, gammas=gamma_grid_12).mae
        print(f"  D1 MAE:    {m1:.4f}")

        d_oracle = np.column_stack([n_in, x_in])
        mo = fit_krr_loocv(d_oracle, v_in, gammas=gamma_grid_12).mae
        print(f"  Oracle MAE:{mo:.4f}")

        results.append({
            "a_left": a_left, "d": d, "delta": a_r - a_left,
            "class": classify_system(a_left, d),
            "splitting": sm, "mae_d1": m1, "mae_oracle": mo,
        })

    splitting = np.array([r["splitting"] for r in results])
    mae_d1 = np.array([r["mae_d1"] for r in results])
    mae_oracle = np.array([r["mae_oracle"] for r in results])
    classes = [r["class"] for r in results]

    rho, pval = spearmanr(splitting, mae_d1)
    print(f"\nSpearman rho (D1): {rho:.3f}, p={pval:.2e}")
    rho_o, pval_o = spearmanr(splitting, mae_oracle)
    print(f"Spearman rho (oracle): {rho_o:.3f}, p={pval_o:.2e}")

    fig, ax = plt.subplots(figsize=(6.5, 5))

    marker_kw = {
        "asymmetry": dict(marker="o", facecolors="none", edgecolors="#d62728", linewidths=1.5, s=70),
        "distance":  dict(marker="o", facecolors="#d62728", edgecolors="#d62728", linewidths=1.0, s=70),
        "both":      dict(marker="o", facecolors="#d62728", edgecolors="black", linewidths=1.5, s=90),
        "cross-grid": dict(marker="s", facecolors="#d62728", edgecolors="#d62728", linewidths=1.0, s=60),
    }
    oracle_kw = {
        "asymmetry": dict(marker="D", facecolors="none", edgecolors="#1f77b4", linewidths=1.5, s=60),
        "distance":  dict(marker="D", facecolors="#1f77b4", edgecolors="#1f77b4", linewidths=1.0, s=60),
        "both":      dict(marker="D", facecolors="#1f77b4", edgecolors="black", linewidths=1.5, s=75),
        "cross-grid": dict(marker="D", facecolors="#1f77b4", edgecolors="#1f77b4", linewidths=1.0, s=50),
    }

    plotted_d1 = set()
    plotted_or = set()
    for i, cls in enumerate(classes):
        label_d1 = None
        if cls not in plotted_d1:
            label_d1 = {
                "asymmetry": r"$D_1=[n]$, $\Delta A$ scan",
                "distance":  r"$D_1=[n]$, $d$ scan",
                "both":      r"$D_1=[n]$, shared",
                "cross-grid": r"$D_1=[n]$, cross-grid",
            }[cls]
            plotted_d1.add(cls)
        ax.scatter(splitting[i], mae_d1[i], label=label_d1, zorder=5, **marker_kw[cls])

        label_or = None
        if cls not in plotted_or:
            label_or = {
                "asymmetry": r"$D_{\mathrm{oracle}}$, $\Delta A$ scan",
                "distance":  r"$D_{\mathrm{oracle}}$, $d$ scan",
                "both":      r"$D_{\mathrm{oracle}}$, shared",
                "cross-grid": r"$D_{\mathrm{oracle}}$, cross-grid",
            }[cls]
            plotted_or.add(cls)
        ax.scatter(splitting[i], mae_oracle[i], label=label_or, zorder=4, **oracle_kw[cls])

    ax.set_xlabel(r"Splitting metric $\langle \sigma_k \rangle$")
    ax.set_ylabel(r"LOO-CV MAE (a.u.)")
    ax.axhline(0, color="gray", linewidth=0.3, linestyle="--")
    ax.text(0.03, 0.92, rf"$D_1$: Spearman $\rho = {rho:.2f}$",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.3))
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9, ncol=1)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = fig_dir / f"fig_5_5_master_plot_expanded.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)

    print("\n" + "=" * 75)
    print(f"{'AL':>5} {'d':>5} {'DeltaA':>7} {'class':>10} "
          f"{'splitting':>10} {'D1 MAE':>10} {'Oracle':>10}")
    print("-" * 75)
    for r in sorted(results, key=lambda r: r["splitting"]):
        print(f"{r['a_left']:5.1f} {r['d']:5.1f} {r['delta']:7.1f} "
              f"{r['class']:>10} {r['splitting']:10.4f} "
              f"{r['mae_d1']:10.4f} {r['mae_oracle']:10.4f}")


if __name__ == "__main__":
    main()
    print("Done.")
