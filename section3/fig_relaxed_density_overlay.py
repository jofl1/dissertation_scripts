import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.constants import rc_params

# Density overlay figure: n^(0)(x) and n^S(x) for two representative S
# values. Top row S=0.4 (the dissertation's headline value), bottom row
# S=0.6 (the largest converged step). Each row pairs a main panel with
# the densities overlaid against a thin lower strip showing Δn(x). The
# point of the figure is that the two curves are visually on top of
# each other.

here = Path(__file__).parent
data_path = here / "results" / "scf_density_vs_S.npz"
fig_dir = here / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

s_panels = [0.4, 0.6]
x_lim = (-12.5, 12.5)


def fmt_sci(val, sig=3):
    if val == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(val))))
    mant = val / 10 ** exp
    return rf"{mant:.{sig-1}f}\times 10^{{{exp}}}"


def main():
    plt.rcParams.update(rc_params)

    with np.load(data_path) as data:
        x = data["x"]
        n_0 = data["n_0_scf"]
        n_panels = {S: data[f"n_S_{S:.4f}"] for S in s_panels}

    fig, axes = plt.subplots(
        nrows=2 * len(s_panels),
        ncols=1,
        figsize=(7.0, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1] * len(s_panels), "hspace": 0.10},
    )

    dx = float(x[1] - x[0])
    right_mask = x > 0

    for i, S in enumerate(s_panels):
        ax_main = axes[2 * i]
        ax_diff = axes[2 * i + 1]
        n_S = n_panels[S]
        delta = n_S - n_0

        dn_max = float(np.max(np.abs(delta)))
        q = 0.5 * float(np.sum(np.abs(delta)) * dx)
        pair_diff = np.abs(n_S - n_S[::-1])
        mirror_dn_max = float(np.max(pair_diff[right_mask]))

        ax_main.plot(x, n_0, color="black", linewidth=1.6,
                     label=r"$n^{(0)}(x)$")
        ax_main.plot(
            x, n_S, color="#d62728", linewidth=1.2, linestyle="--",
            label=rf"$n^{{S}}(x),\ \mathrm{{step}}\ S = {S}$ a.u.",
        )
        ax_main.set_ylabel(r"$n(x)$ (a.u.)")
        ax_main.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax_main.set_xlim(*x_lim)
        annotation = (
            rf"$\|\Delta n\|_\infty = {fmt_sci(dn_max)}$"
            "\n"
            rf"$Q(S) = {fmt_sci(q)}$"
            "\n"
            rf"$\delta n_{{\mathrm{{mirror}}}}^{{\max}} = {fmt_sci(mirror_dn_max)}$"
        )
        ax_main.text(
            0.02, 0.92, annotation,
            transform=ax_main.transAxes, fontsize=8.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="lightgray", alpha=0.85),
        )

        # Residual panel: scale Δn by an explicit power of 10 so the y-axis
        # label carries the magnitude (replaces matplotlib's small offset
        # text that is easy to miss).
        exponent = int(math.floor(math.log10(dn_max)))
        scale = 10 ** exponent
        ax_diff.plot(x, delta / scale, color="#d62728", linewidth=1.2)
        ax_diff.axhline(0, color="gray", linewidth=0.3, linestyle="--")
        ax_diff.set_ylabel(rf"$\Delta n(x)\ /\ 10^{{{exponent}}}$")
        ax_diff.set_xlim(*x_lim)
        ymax_scaled = 1.05 * dn_max / scale
        ax_diff.set_ylim(-ymax_scaled, ymax_scaled)

    axes[-1].set_xlabel(r"$x$ (a.u.)")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = fig_dir / f"fig_relaxed_density_overlay.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
    print("Done.")
