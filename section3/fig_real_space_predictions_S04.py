from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.constants import rc_params, trim_margin
from utils.krr import fit_krr_loocv
from utils.toy_target import make_toy_target

# Real-space KRR predictions at S = 0.4 for D_1 and the oracle. The local
# D_1 model is forced to be a function of n alone, so on a symmetric
# baseline density it has to collapse to the branch average. The oracle
# (n, x) recovers the antisymmetric step.

here = Path(__file__).parent
fig_dir = here / "figures"
data_dir = here / "data"
res_dir = here / "results"
baseline_path = data_dir / "baseline_symmetric_uu_d10.npz"
fig_dir.mkdir(parents=True, exist_ok=True)
res_dir.mkdir(parents=True, exist_ok=True)

s_demo = 0.4


def main():
    plt.rcParams.update(rc_params)

    with np.load(baseline_path) as baseline:
        x = baseline["x"]
        n = baseline["n"]
        v_xc = baseline["v_xc"]

    margin = trim_margin(len(x))
    sl = slice(margin, -margin)
    x_inner = x[sl]
    n_inner = n[sl]
    v_toy = make_toy_target(v_xc, x, s_demo)
    y_exact = v_toy[sl]

    d1_label = r"$D_1 = [n]$"
    oracle_label = r"$D_{\mathrm{oracle}} = [n, x]$"
    descriptors = {
        d1_label: n_inner.reshape(-1, 1),
        oracle_label: np.column_stack([n_inner, x_inner]),
    }
    styles = {
        d1_label: {"color": "#d62728", "linestyle": "--", "linewidth": 2},
        oracle_label: {"color": "#1f77b4", "linestyle": "-", "linewidth": 1.8},
    }

    predictions = {}
    for name, X in descriptors.items():
        print(f"  Grid-searching {name} ...")
        res = fit_krr_loocv(X, y_exact)
        predictions[name] = res.predictions
        print(f"    alpha={res.alpha:.1e}, gamma={res.gamma:.1e}, MAE={res.mae:.6f}")

    fig, (ax_pred, ax_res) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax_pred.plot(x_inner, y_exact, color="black", linewidth=2.5,
                 label=r"Exact $v_{\mathrm{toy}}(x)$", zorder=5)
    for name in descriptors:
        ax_pred.plot(x_inner, predictions[name], label=name, **styles[name])
    ax_pred.axhline(0, color="gray", linewidth=0.3, linestyle="--")
    ax_pred.set_ylabel(r"$v_{\mathrm{toy}}(x)$ (a.u.)")
    ax_pred.legend(loc="lower right", fontsize=9)
    ax_pred.text(0.03, 0.92, "(a)", transform=ax_pred.transAxes, fontsize=12,
                 fontweight="bold")

    # Bottom panel: residuals. The D_1 residual is antisymmetric with
    # magnitude ~S/2 across each well -- the branch-averaging signature.
    for name in descriptors:
        residual = predictions[name] - y_exact
        ax_res.plot(x_inner, residual, label=name, **styles[name])
    ax_res.axhline(0, color="black", linewidth=0.8)
    ax_res.set_xlabel(r"$x$ (a.u.)")
    ax_res.set_ylabel(r"Residual (a.u.)")
    ax_res.legend(loc="lower right", fontsize=9)
    ax_res.text(0.03, 0.88, "(b)", transform=ax_res.transAxes, fontsize=12,
                fontweight="bold")

    fig.suptitle(rf"Real-space KRR predictions ($S = {s_demo}$)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ("pdf", "png"):
        path = fig_dir / f"fig_real_space_predictions_S04.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)

    np.savez(
        res_dir / "real_space_predictions_S04.npz",
        x=x_inner, y_exact=y_exact,
        pred_D1=predictions[d1_label],
        pred_oracle=predictions[oracle_label],
    )
    print(f"Saved: {res_dir / 'real_space_predictions_S04.npz'}")


if __name__ == "__main__":
    main()
    print("Done.")
