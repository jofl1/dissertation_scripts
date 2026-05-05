from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.constants import (
    a_r, d_default, gamma_grid_12, k_neighbours, rc_params, trim_margin,
)
from utils.krr import fit_krr_loocv
from utils.metrics import splitting_metric_knn

# Splitting metric and KRR MAE vs DeltaA across the 5-system asymmetry
# scan at d=10. Both grow with DeltaA -- the local model is constrained
# to a single-valued curve in (n, v_xc), so the more the projection
# bifurcates, the worse it gets.

here = Path(__file__).parent
data_dir = here / "data"
fig_dir = here / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

all_als = [1.0, 1.2, 1.5, 1.8, 2.0]


def main():
    plt.rcParams.update(rc_params)

    deltas = []
    splitting = []
    mae_d1 = []
    mae_oracle = []

    for a_left in all_als:
        print(f"Processing AL={a_left} ...")
        path = data_dir / f"exact_asymmetric_uu_AL{a_left}_AR{a_r}_d{d_default}.npz"
        with np.load(path) as data:
            x = data["x"]
            n = data["n"]
            v_xc_raw = data["v_xc"]
        v_xc = v_xc_raw - v_xc_raw[-1]

        sl = slice(trim_margin(len(x)), -trim_margin(len(x)))
        x_in, n_in, v_in = x[sl], n[sl], v_xc[sl]

        deltas.append(a_r - a_left)

        sm = splitting_metric_knn(n_in, v_in, k=k_neighbours)
        splitting.append(sm)
        print(f"  Splitting metric: {sm:.4f}")

        m1 = fit_krr_loocv(n_in.reshape(-1, 1), v_in, gammas=gamma_grid_12).mae
        mae_d1.append(m1)
        print(f"  D1 MAE: {m1:.4f}")

        d_oracle = np.column_stack([n_in, x_in])
        mo = fit_krr_loocv(d_oracle, v_in, gammas=gamma_grid_12).mae
        mae_oracle.append(mo)
        print(f"  Oracle MAE: {mo:.4f}")

    deltas = np.array(deltas)
    splitting = np.array(splitting)
    mae_d1 = np.array(mae_d1)
    mae_oracle = np.array(mae_oracle)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(deltas, splitting, "o-", color="#d62728", linewidth=1.8,
             markersize=8, label=r"$\langle \sigma_k \rangle$")
    ax1.set_xlabel(r"Well-depth asymmetry $\Delta A = A_R - A_L$ (a.u.)")
    ax1.set_ylabel(r"Splitting metric $\langle \sigma_k \rangle$")
    ax1.axhline(0, color="gray", linewidth=0.3, linestyle="--")
    ax1.text(0.03, 0.92, r"$\mathbf{(a)}$", transform=ax1.transAxes,
             fontsize=12, fontweight="bold", zorder=10,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.3))

    ax2.plot(deltas, mae_d1, "o-", color="#d62728", linewidth=1.8,
             markersize=8, label=r"$D_1 = [n]$")
    ax2.plot(deltas, mae_oracle, "D--", color="#1f77b4", linewidth=1.8,
             markersize=8, label=r"$D_{\mathrm{oracle}} = [n, x]$")
    ax2.set_xlabel(r"Well-depth asymmetry $\Delta A = A_R - A_L$ (a.u.)")
    ax2.set_ylabel(r"LOO-CV MAE (a.u.)")
    ax2.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax2.axhline(0, color="gray", linewidth=0.3, linestyle="--")
    ax2.text(0.03, 0.92, r"$\mathbf{(b)}$", transform=ax2.transAxes,
             fontsize=12, fontweight="bold", zorder=10,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.3))

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = fig_dir / f"fig_5_3b_splitting_vs_asymmetry.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
    print("Done.")
