from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.constants import (
    a_r, gamma_grid_10, gamma_grid_12, rc_params, trim_margin,
)
from utils.descriptors import build_window
from utils.krr import fit_krr_loocv

# Window-descriptor MAE vs half-width ell for three representative exact
# systems. Once the window spans a few bohr, the local model recovers the
# missing nonlocal context and MAE collapses towards the oracle level.
# Uses the original 10-element gamma grid for window descriptors (kept to
# preserve the published numbers); oracle dotted lines use the 12-element
# grid to match the rest of the section4 figures.

here = Path(__file__).parent
data_dir = here / "data"
fig_dir = here / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

window_half_widths = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]

systems = [
    (1.0, 10.0, r"$\Delta A=1.0,\; d=10$"),
    (1.0, 6.0,  r"$\Delta A=1.0,\; d=6$"),
    (1.5, 10.0, r"$\Delta A=0.5,\; d=10$"),
]


def load_system(a_left, d):
    path = data_dir / f"exact_asymmetric_uu_AL{a_left}_AR{a_r}_d{d}.npz"
    with np.load(path) as data:
        x = data["x"].copy()
        n = data["n"].copy()
        v_xc_raw = data["v_xc"].copy()
    v_xc = v_xc_raw - v_xc_raw[-1]
    return x, n, v_xc


def main():
    plt.rcParams.update(rc_params)

    fig, ax = plt.subplots(figsize=(6.5, 5))

    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    markers = ["o", "s", "D"]

    for idx, (a_left, d, label) in enumerate(systems):
        print(f"\n{'=' * 60}")
        print(f"System: AL={a_left}, d={d}, DeltaA={a_r - a_left}")
        print(f"{'=' * 60}")

        x, n, v_xc = load_system(a_left, d)

        maes = []
        for ell in window_half_widths:
            print(f"  ell={ell:.1f} ...", end=" ", flush=True)
            indices, X_window = build_window(x, n, ell)
            if indices is None:
                maes.append(np.nan)
                print("SKIPPED")
                continue
            y = v_xc[indices]
            mae = fit_krr_loocv(X_window, y, gammas=gamma_grid_10).mae
            maes.append(mae)
            print(f"MAE={mae:.4f}")

        ax.plot(window_half_widths, maes, f"{markers[idx]}-",
                color=colors[idx], linewidth=1.8, markersize=8,
                label=label)

    # Oracle reference lines on the boundary-trimmed grid.
    for idx, (a_left, d, _) in enumerate(systems):
        x, n, v_xc = load_system(a_left, d)
        sl = slice(trim_margin(len(x)), -trim_margin(len(x)))
        x_in, n_in, v_in = x[sl], n[sl], v_xc[sl]
        d_oracle = np.column_stack([n_in, x_in])
        oracle_mae = fit_krr_loocv(d_oracle, v_in, gammas=gamma_grid_12).mae
        ax.axhline(oracle_mae, color=colors[idx], linewidth=0.8,
                   linestyle=":", alpha=0.5)
        print(f"\nOracle MAE for AL={systems[idx][0]}, d={systems[idx][1]}: {oracle_mae:.4f}")

    ax.set_xlabel(r"Window half-width $\ell$ (a.u.)")
    ax.set_ylabel(r"LOO-CV MAE (a.u.)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.axhline(0, color="gray", linewidth=0.3, linestyle="--")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = fig_dir / f"fig_5_6_window_scan_exact.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
    print("Done.")
