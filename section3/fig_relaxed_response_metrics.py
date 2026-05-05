import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.constants import rc_params

# Density-response measures vs S on a log y-axis. Plots Q(S),
# δn_mirror^max(S), |ΔN_L|(S), and ‖Δn‖∞(S) together. No "y = S"
# reference line -- that would mix an energy with a density-norm on a
# single axis. The annotation reports max Q, min S/Q, and max
# δn_mirror^max for the dissertation headline.

here = Path(__file__).parent
csv_path = here / "results" / "response_metrics_vs_S.csv"
fig_dir = here / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)


def fmt_sci(val, sig=3):
    if val == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(val))))
    mant = val / 10 ** exp
    return rf"{mant:.{sig-1}f}\times 10^{{{exp}}}"


def main():
    plt.rcParams.update(rc_params)

    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    rows.sort(key=lambda r: r["S"])
    cols = {k: np.array([r[k] for r in rows]) for k in rows[0]}

    # Drop S = 0; the response measures collapse to zero there and would
    # break the log y-axis.
    nz = cols["S"] > 0
    cols = {k: v[nz] for k, v in cols.items()}
    S = cols["S"]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))

    plots = [
        ("Q", r"$Q(S)=\frac{1}{2}\|n^S-n^{(0)}\|_1$", "o", "#d62728"),
        ("mirror_dn_max", r"$\delta n_{\mathrm{mirror}}^{\max}(S)$", "s", "#1f77b4"),
        ("dN_left", r"$|\Delta N_L|(S)$", "^", "#ff7f0e"),
        ("dn_Linf", r"$\|\Delta n\|_\infty(S)$", "v", "#2ca02c"),
    ]
    for key, label, marker, color in plots:
        ax.plot(S, np.abs(cols[key]), f"{marker}-", color=color,
                linewidth=1.4, label=label)

    ax.set_yscale("log")
    ax.set_xlabel(r"Imposed step height $S$ (a.u.)")
    ax.set_ylabel("Density-response measure")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    if S.size > 0:
        max_q = float(cols["Q"].max())
        min_step_per_q = float(cols["step_per_Q"].min())
        max_mirror = float(cols["mirror_dn_max"].max())
        text = (
            rf"$\max\,Q(S) = {fmt_sci(max_q)}$"
            "\n"
            rf"$\min_{{S>0}}\,S/Q(S) = {fmt_sci(min_step_per_q)}$ Ha/e"
            "\n"
            rf"$\max\,\delta n_{{\mathrm{{mirror}}}}^{{\max}} = {fmt_sci(max_mirror)}$"
        )
        ax.text(0.02, 0.97, text, transform=ax.transAxes, fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="lightgray", alpha=0.85))

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = fig_dir / f"fig_relaxed_response_metrics.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
    print("Done.")
