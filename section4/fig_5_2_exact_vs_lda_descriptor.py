from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.constants import a_r, d_default, rc_params, trim_margin

# Two-panel descriptor-space scatter for the representative system. Panel
# (a): exact v_xc vs n(x) splits into red/blue branches by position.
# Panel (b): LDA v_xc collapses to a single-valued curve. Same density
# is fed to both -- the split comes from the functional form, not the data.

here = Path(__file__).parent
data_dir = here / "data"
fig_dir = here / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

a_left = 1.0
data_path = data_dir / f"exact_asymmetric_uu_AL{a_left}_AR{a_r}_d{d_default}.npz"


def main():
    plt.rcParams.update(rc_params)

    with np.load(data_path) as data:
        x = data["x"]
        n = data["n"]
        v_xc = data["v_xc"]
        v_xc_lda = data["v_xc_lda"]

    sl = slice(trim_margin(len(x)), -trim_margin(len(x)))
    x_in = x[sl]
    n_in = n[sl]
    v_xc_in = v_xc[sl]
    v_lda_in = v_xc_lda[sl]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    scatter_kw = dict(
        cmap="coolwarm", s=25, alpha=0.85, edgecolors="none",
        vmin=x_in.min(), vmax=x_in.max(),
    )

    sc = ax1.scatter(n_in, v_xc_in, c=x_in, **scatter_kw)
    ax1.set_xlabel(r"$n(x)$")
    ax1.set_ylabel(r"$v_{\mathrm{xc}}(x)$ (a.u.)")
    ax1.set_title(r"Exact $v_{\mathrm{xc}}$", fontsize=12)
    ax1.axhline(0, color="gray", linewidth=0.3, linestyle="--")
    ax1.text(0.03, 0.92, r"$\mathbf{(a)}$", transform=ax1.transAxes,
             fontsize=12, fontweight="bold", zorder=10,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.3))

    ax2.scatter(n_in, v_lda_in, c=x_in, **scatter_kw)
    ax2.set_xlabel(r"$n(x)$")
    ax2.set_title(r"LDA $v_{\mathrm{xc}}$", fontsize=12)
    ax2.axhline(0, color="gray", linewidth=0.3, linestyle="--")
    ax2.text(0.03, 0.92, r"$\mathbf{(b)}$", transform=ax2.transAxes,
             fontsize=12, fontweight="bold", zorder=10,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.3))

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    fig.colorbar(sc, cax=cax, label=r"$x$ (a.u.)")

    for ext in ("pdf", "png"):
        path = fig_dir / f"fig_5_2_exact_vs_lda_descriptor.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
    print("Done.")
