from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.constants import a_r, d_default, rc_params, trim_margin

# Three-panel real-space view of the representative AL=1.0, AR=2.0, d=10
# system: external potential, exact density, exact v_xc. Reads the cached
# baseline written by make_baseline.py.

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
        v_ext = data["v_ext"]
        n = data["n"]
        v_xc = data["v_xc"]

    sl = slice(trim_margin(len(x)), -trim_margin(len(x)))

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

    ax = axes[0]
    ax.plot(x, v_ext, color="black", linewidth=1.5)
    ax.set_ylabel(r"$v_{\mathrm{ext}}(x)$ (a.u.)")
    ax.text(0.03, 0.90, "(a)", transform=ax.transAxes, fontsize=12,
            fontweight="bold")

    ax = axes[1]
    ax.plot(x, n, color="steelblue", linewidth=1.5)
    ax.fill_between(x, 0, n, alpha=0.15, color="steelblue")
    ax.set_ylabel(r"$n(x)$ (a.u.)")
    ax.text(0.03, 0.90, "(b)", transform=ax.transAxes, fontsize=12,
            fontweight="bold")

    ax = axes[2]
    ax.plot(x[sl], v_xc[sl], color="#d62728", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.3, linestyle="--")
    ax.set_ylabel(r"$v_{xc}(x)$ (a.u.)")
    ax.set_xlabel(r"$x$ (a.u.)")
    ax.text(0.03, 0.90, "(c)", transform=ax.transAxes, fontsize=12,
            fontweight="bold")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = fig_dir / f"fig_5_1_real_space_exact.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
    print("Done.")
