# Step heights for the toy family. Mapped from the old A range as S = 2*A
# so absolute step heights match the old curves.
s_values = [0.0, 0.04, 0.10, 0.20, 0.40, 1.00]

# Step heights for the constrained-KS response sweep (sec 3.4). S=1.0 is
# dropped because the SCF enters a charge-transfer instability between
# S~0.66 and S~0.74; S=0.60 sits inside the converged regime and supports
# the same headline message as S=0.40 with one extra point.
s_values_response = [0.0, 0.04, 0.10, 0.20, 0.40, 0.60]


def trim_margin(n_points):
    # max(3, N // 20) -- 5% trim with a hard floor at 3 grid points.
    return max(3, n_points // 20)


rc_params = {
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
}
