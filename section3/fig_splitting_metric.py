import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.constants import rc_params, s_values, trim_margin
from utils.metrics import splitting_metric_knn
from utils.toy_target import make_toy_target

# KNN splitting metric vs S for the toy family. Predicted scaling is
# <sigma_k> ~ S/2, with the slope falling slightly short of 1/2 because
# the six-point neighbourhoods (point + 5 NN) are rarely a perfect 3-3
# left/right split, and the central x=0 point is unshifted (sgn(0) = 0).

here = Path(__file__).parent
fig_dir = here / "figures"
data_dir = here / "data"
res_dir = here / "results"
baseline_path = data_dir / "baseline_symmetric_uu_d10.npz"
fig_dir.mkdir(parents=True, exist_ok=True)
res_dir.mkdir(parents=True, exist_ok=True)


def main():
    plt.rcParams.update(rc_params)

    with np.load(baseline_path) as baseline:
        x = baseline["x"]
        n = baseline["n"]
        v_xc = baseline["v_xc"]

    margin = trim_margin(len(x))
    sl = slice(margin, -margin)
    n_inner = n[sl]

    metrics = []
    for S in s_values:
        v_toy = make_toy_target(v_xc, x, S)
        sigma = splitting_metric_knn(n_inner, v_toy[sl])
        metrics.append(sigma)
        print(f"  S = {S:.2f}  →  <sigma_k> = {sigma:.6f}")

    s_arr = np.array(s_values, dtype=float)
    sigma_arr = np.array(metrics, dtype=float)

    # Linear fit excluding S = 0 -- the floor at S = 0 is the within-well
    # ambiguity, not part of the linear S/2 trend, so including it would
    # bias the slope.
    mask = s_arr > 0
    slope, intercept = np.polyfit(s_arr[mask], sigma_arr[mask], 1)
    fit_pred = slope * s_arr + intercept
    ss_res = np.sum((sigma_arr - fit_pred) ** 2)
    ss_tot = np.sum((sigma_arr - sigma_arr.mean()) ** 2)
    if ss_tot > 0:
        r2 = 1 - ss_res / ss_tot
    else:
        r2 = 0.0

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(s_arr, sigma_arr, "o-", color="#d62728", linewidth=2,
            markersize=8, markeredgecolor="black", markeredgewidth=0.8,
            zorder=5, label="KNN splitting metric")
    s_fit = np.linspace(0, s_arr.max() * 1.05, 100)
    ax.plot(s_fit, slope * s_fit + intercept, "--", color="gray", linewidth=1.5,
            alpha=0.7, label=rf"Linear fit (slope $= {slope:.3f}$)")
    ax.text(0.05, 0.88, rf"$R^2 = {r2:.4f}$", transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.9))
    ax.set_xlabel(r"Step height $S$ (a.u.)")
    ax.set_ylabel(r"Average splitting metric $\langle \sigma_k \rangle$")
    ax.set_xlim(-0.02, s_arr.max() * 1.1)
    ax.set_ylim(bottom=-0.005)
    ax.axhline(0, color="gray", linewidth=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = fig_dir / f"fig_splitting_metric.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)

    out_csv = res_dir / "splitting_metric_vs_S.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["S", "sigma_k", "fit_slope", "fit_intercept", "fit_r2"])
        for S, sigma in zip(s_arr, sigma_arr):
            w.writerow([S, sigma, slope, intercept, r2])
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
    print("Done.")
