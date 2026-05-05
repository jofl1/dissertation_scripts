from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.constants import a_r, d_default, rc_params, trim_margin

# Four-panel descriptor-space scatter at d=10, varying the left-well depth
# AL through {1.0, 1.5, 1.8, 2.0}. As DeltaA = AR - AL grows the (n, v_xc)
# projection bifurcates into red/blue branches.

here = Path(__file__).parent
data_dir = here / "data"
fig_dir = here / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

representative_als = [1.0, 1.5, 1.8, 2.0]


def main():
    plt.rcParams.update(rc_params)

    n_panels = len(representative_als)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 4.5),
                             sharey=True)

    scatter_kw = dict(cmap="coolwarm", s=20, alpha=0.85, edgecolors="none")

    for i, a_left in enumerate(representative_als):
        ax = axes[i]
        path = data_dir / f"exact_asymmetric_uu_AL{a_left}_AR{a_r}_d{d_default}.npz"
        with np.load(path) as data:
            x = data["x"]
            n = data["n"]
            v_xc_raw = data["v_xc"]
        v_xc = v_xc_raw - v_xc_raw[-1]

        sl = slice(trim_margin(len(x)), -trim_margin(len(x)))
        x_in, n_in, v_in = x[sl], n[sl], v_xc[sl]

        sc = ax.scatter(n_in, v_in, c=x_in,
                        vmin=x_in.min(), vmax=x_in.max(), **scatter_kw)
        ax.axhline(0, color="gray", linewidth=0.3, linestyle="--")
        ax.set_xlabel(r"$n(x)$")

        delta = a_r - a_left
        ax.set_title(rf"$A_L={a_left:.1f}$" + "\n" + rf"$\Delta A={delta:.1f}$",
                     fontsize=11)

        label = chr(ord("a") + i)
        ax.text(0.03, 0.92, rf"$\mathbf{{({label})}}$",
                transform=ax.transAxes, fontsize=12, fontweight="bold",
                zorder=10,
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.7, pad=0.3))

    axes[0].set_ylabel(r"$v_{\mathrm{xc}}(x)$ (a.u.)")

    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    fig.colorbar(sc, cax=cax, label=r"$x$ (a.u.)")

    for ext in ("pdf", "png"):
        path = fig_dir / f"fig_5_3a_asymmetry_scatter.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
    print("Done.")
