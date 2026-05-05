import numpy as np

# Density-response and mirror-pair metrics for one S value. dn_L1 and Q
# are integrated electron-number displacements (dimensionless). S is in
# atomic units (Ha), so S/Q has units Ha per displaced electron and is a
# response coefficient, not a dimensionless ratio. mirror_dn_max measures
# whether two points symmetric about x=0 still have similar density inputs
# after relaxation while their target potentials still differ by exactly S.


def compute_metrics(x, dx, n_0, n_S, S):
    delta = n_S - n_0
    dn_L1 = float(np.sum(np.abs(delta)) * dx)
    dn_Linf = float(np.max(np.abs(delta)))
    dN_left = float(np.sum(delta[x < 0]) * dx)

    q = 0.5 * dn_L1
    step_per_q = float(S / q) if q > 0 else float("inf")

    pair_diff = np.abs(n_S - n_S[::-1])[x > 0]
    mirror_dn_max = float(np.max(pair_diff))
    mirror_dn_mean = float(np.mean(pair_diff))
    step_per_mirror_dn_max = (
        float(S / mirror_dn_max) if mirror_dn_max > 0 else float("inf")
    )

    return {
        "S": float(S),
        "dn_L1": dn_L1,
        "dn_Linf": dn_Linf,
        "dN_left": dN_left,
        "Q": float(q),
        "step_per_Q": step_per_q,
        "mirror_dn_max": mirror_dn_max,
        "mirror_dn_mean": mirror_dn_mean,
        "step_per_mirror_dn_max": step_per_mirror_dn_max,
    }
