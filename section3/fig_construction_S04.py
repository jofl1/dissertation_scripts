from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.constants import rc_params, trim_margin
from utils.toy_target import make_toy_target

# Construction figure: baseline v_xc^(0), the antisymmetric (S/2)*sgn(x)
# step, and the resulting v_toy at S = 0.4.

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
        v_xc = baseline["v_xc"]

    # Trim vacuum tails so the figure is centred on the wells.
    margin = trim_margin(len(x))
    sl = slice(margin, -margin)
    x_inner = x[sl]
    v_xc_inner = v_xc[sl]

    step = 0.5 * s_demo * np.sign(x_inner)
    v_toy = make_toy_target(v_xc_inner, x_inner, s_demo)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(x_inner, v_xc_inner, color="black", linewidth=1.5,
            label=r"$v_{xc}^{(0)}(x)$ (baseline)")
    ax.plot(x_inner, step, color="steelblue", linewidth=1.5, linestyle="--",
            label=rf"$(S/2)\,\mathrm{{sgn}}(x)$,  $S={s_demo}$")
    ax.plot(x_inner, v_toy, color="#d62728", linewidth=1.5,
            label=r"$v_{\mathrm{toy}}(x) = v_{xc}^{(0)} + (S/2)\,\mathrm{sgn}(x)$")
    ax.axhline(0, color="gray", linewidth=0.3, linestyle="--")
    ax.set_ylabel(r"$V$ (a.u.)")
    ax.set_xlabel(r"$x$ (a.u.)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = fig_dir / f"fig_construction_S04.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
    print("Done.")
