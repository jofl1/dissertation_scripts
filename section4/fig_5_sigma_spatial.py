from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from utils.constants import a_r, d_default, k_neighbours, rc_params, trim_margin
from utils.metrics import per_point_sigma

# Per-point splitting metric sigma_k(x) for three systems at d=10 with
# DeltaA = 0, 0.5, 1.0. Plots raw (no smoothing) to expose the within-well
# ambiguity floor present even at DeltaA=0; the inter-well overlap that
# appears with growing asymmetry sits on top of that floor.

here = Path(__file__).parent
data_dir = here / "data"
fig_dir = here / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

systems = [
    (2.0, r"$\Delta A = 0$"),
    (1.5, r"$\Delta A = 0.5$"),
    (1.0, r"$\Delta A = 1.0$"),
]


def main():
    plt.rcParams.update(rc_params)

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.5), sharex=True)

    panel_data = []
    for a_left, label in systems:
        path = data_dir / f"exact_asymmetric_uu_AL{a_left}_AR{a_r}_d{d_default}.npz"
        with np.load(path) as data:
            x = data["x"].copy()
            n = data["n"].copy()
            v_xc_raw = data["v_xc"].copy()
        v_xc = v_xc_raw - v_xc_raw[-1]

        sl = slice(trim_margin(len(x)), -trim_margin(len(x)))
        x_in, n_in, v_in = x[sl], n[sl], v_xc[sl]
        sigma = per_point_sigma(n_in, v_in, k=k_neighbours)
        panel_data.append((x_in, n_in, sigma, label))

    sigma_max = max(np.max(d[2]) for d in panel_data) * 1.15

    for idx, (x_in, n_in, sigma, label) in enumerate(panel_data):
        ax = axes[idx]
        ax.plot(x_in, sigma, color="#d62728", linewidth=1.0,
                label=r"$\sigma_k(x)$", zorder=3)
        ax.set_ylabel(r"$\sigma_k(x)$ (a.u.)")
        ax.set_ylim(-0.003, sigma_max)

        ax2 = ax.twinx()
        ax2.fill_between(x_in, n_in, alpha=0.12, color="#1f77b4")
        ax2.plot(x_in, n_in, color="#1f77b4", linewidth=0.8, alpha=0.4)
        ax2.set_ylabel(r"$n(x)$ (a.u.)", color="#1f77b4", alpha=0.7)
        ax2.tick_params(axis="y", labelcolor="#1f77b4", colors="#1f77b4")
        ax2.set_ylim(bottom=0)

        ax.axvline(-d_default / 2, color="gray", linewidth=0.5, linestyle=":",
                   alpha=0.5, zorder=1)
        ax.axvline(d_default / 2, color="gray", linewidth=0.5, linestyle=":",
                   alpha=0.5, zorder=1)
        ax.axvspan(-1.5, 1.5, alpha=0.06, color="gray", zorder=0)

        panel = "abc"[idx]
        ax.text(0.02, 0.92, f"({panel}) {label}",
                transform=ax.transAxes, fontsize=11,
                verticalalignment="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))

        print(f"{label}: <sigma_k> = {np.mean(sigma):.4f}, "
              f"max sigma_k = {np.max(sigma):.4f} at x = {x_in[np.argmax(sigma)]:.1f}")

        if idx == 0:
            lines1, _ = ax.get_legend_handles_labels()
            density_patch = Patch(facecolor="#1f77b4", alpha=0.12,
                                  edgecolor="#1f77b4", label=r"$n(x)$")
            barrier_patch = Patch(facecolor="gray", alpha=0.06,
                                  edgecolor="none", label="barrier region")
            ax.legend(handles=lines1 + [density_patch, barrier_patch],
                      loc="upper right", fontsize=8, framealpha=0.9)

    axes[-1].set_xlabel(r"Position $x$ (a.u.)")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = fig_dir / f"fig_5_sigma_spatial.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
    print("Done.")
