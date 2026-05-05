from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.constants import rc_params, trim_margin
from utils.toy_target import make_toy_target

# Matched-pair figure at S = 0.4. Real space and descriptor space side by
# side, with a labelled mirror pair L (x < 0) and R (x > 0). Because the
# baseline density is symmetric and the step is exactly antisymmetric,
# every off-centre mirror pair has identical n and Delta v_toy = S.

here = Path(__file__).parent
fig_dir = here / "figures"
data_dir = here / "data"
baseline_path = data_dir / "baseline_symmetric_uu_d10.npz"
fig_dir.mkdir(parents=True, exist_ok=True)

s_demo = 0.4


def main():
    plt.rcParams.update(rc_params)

    with np.load(baseline_path) as baseline:
        x = baseline["x"]
        n = baseline["n"]
        v_xc = baseline["v_xc"]
    v_toy = make_toy_target(v_xc, x, s_demo)

    margin = trim_margin(len(x))
    sl = slice(margin, -margin)
    x_inner = x[sl]
    n_inner = n[sl]
    v_inner = v_toy[sl]

    # Pick a clean mirror pair near the well centres at x = +/-5, where the
    # density is large enough to read the matched n value confidently.
    iR = np.argmin(np.abs(x - 5.0))
    iL = np.argmin(np.abs(x + 5.0))
    assert np.isclose(n[iL], n[iR], atol=1e-9), "density should be symmetric at the wells"

    pair_points = [(iL, "o", "L"), (iR, "s", "R")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(x_inner, v_inner, color="#d62728", linewidth=1.5,
             label=r"$v_{\mathrm{toy}}(x)$")
    # Density rescaled and offset so it sits cleanly under v_toy on the
    # same axes -- only the shape matters here, not the absolute scale.
    ax1.plot(x_inner, n_inner * 2 - 0.8, color="steelblue", linewidth=1.2,
             linestyle="--", label=r"$n(x)$ (scaled)")
    for idx_pt, marker, lbl in pair_points:
        ax1.plot(x[idx_pt], v_toy[idx_pt], marker, color="black", markersize=10,
                 markeredgewidth=2, markerfacecolor="none", zorder=10)
        ax1.annotate(lbl, (x[idx_pt], v_toy[idx_pt]),
                     textcoords="offset points", xytext=(8, 8),
                     fontsize=11, fontweight="bold")
    ax1.axhline(0, color="gray", linewidth=0.3, linestyle="--")
    ax1.set_xlabel(r"$x$ (a.u.)")
    ax1.set_ylabel(r"$v_{\mathrm{toy}}(x)$ (a.u.)")
    ax1.set_title(rf"Real space ($S = {s_demo}$)", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.text(0.03, 0.90, "(a)", transform=ax1.transAxes, fontsize=12,
             fontweight="bold")

    # Descriptor-space panel. Shuffle the colour-coded points so neither
    # branch sits entirely on top of the other in the scatter.
    rng = np.random.default_rng(42)
    idx_perm = rng.permutation(len(x_inner))
    sc = ax2.scatter(n_inner[idx_perm], v_inner[idx_perm],
                     c=x_inner[idx_perm], cmap="coolwarm", s=14, alpha=0.7,
                     edgecolors="none")
    for idx_pt, marker, lbl in pair_points:
        ax2.plot(n[idx_pt], v_toy[idx_pt], marker, color="black", markersize=10,
                 markeredgewidth=2, markerfacecolor="none", zorder=10)
        ax2.annotate(lbl, (n[idx_pt], v_toy[idx_pt]),
                     textcoords="offset points", xytext=(8, 8),
                     fontsize=11, fontweight="bold")
    ax2.annotate("", xy=(n[iR], v_toy[iR]), xytext=(n[iL], v_toy[iL]),
                 arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.5))
    mid_n = 0.5 * (n[iL] + n[iR])
    mid_v = 0.5 * (v_toy[iL] + v_toy[iR])
    dv = v_toy[iR] - v_toy[iL]
    ax2.text(mid_n + 0.02, mid_v, rf"$\Delta v_{{\mathrm{{toy}}}} = {dv:.3f}$",
             fontsize=10, ha="left")
    ax2.axhline(0, color="gray", linewidth=0.3, linestyle="--")
    ax2.set_xlabel(r"$n(x)$")
    ax2.set_ylabel(r"$v_{\mathrm{toy}}(x)$ (a.u.)")
    ax2.set_title(rf"Descriptor space ($S = {s_demo}$)", fontsize=11)
    plt.colorbar(sc, ax=ax2, label=r"$x$ (a.u.)", shrink=0.85)
    ax2.text(0.03, 0.90, "(b)", transform=ax2.transAxes, fontsize=12,
             fontweight="bold")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = fig_dir / f"fig_matched_pair_S04.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)

    # Print sanity-check numbers -- confirms the mirror pair really is
    # identical in n and separated by exactly S in v_toy.
    print(f"Matched pair: x_L={x[iL]:.3f}, x_R={x[iR]:.3f}")
    print(f"  n(x_L) = {n[iL]:.6f}, n(x_R) = {n[iR]:.6f}, |dn| = {abs(n[iL]-n[iR]):.2e}")
    print(f"  v_toy(x_L) = {v_toy[iL]:.6f}, v_toy(x_R) = {v_toy[iR]:.6f}")
    print(f"  dv_toy = v_toy(x_R) - v_toy(x_L) = {dv:.6f}  (expected: {s_demo})")


if __name__ == "__main__":
    main()
    print("Done.")
