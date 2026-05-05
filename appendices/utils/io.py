import re
from pathlib import Path

import numpy as np

from utils.constants import a_r, trim_margin

# I/O helpers shared by the appendix scripts. discover_systems globs
# section4's data dir for all 19 exact systems; load_full and load_inner
# read one npz, gauge-pin v_xc to zero at the right boundary, and return
# either the full grid or the boundary-trimmed inner slice.

filename_pattern = re.compile(
    r"exact_asymmetric_uu_AL([\d.]+)_AR2\.0_d([\d.]+)\.npz$"
)


def discover_systems(data_dir):
    systems = []
    for f in sorted(Path(data_dir).glob("exact_asymmetric_uu_AL*_AR2.0_d*.npz")):
        # Skip earlier L=50 box-size test files if any are lying around.
        if "_L" in f.name:
            continue
        m = filename_pattern.match(f.name)
        if m:
            systems.append((float(m.group(1)), float(m.group(2))))
    return systems


def load_full(data_dir, a_left, d):
    path = Path(data_dir) / f"exact_asymmetric_uu_AL{a_left}_AR{a_r}_d{d}.npz"
    with np.load(path) as data:
        x = data["x"].copy()
        n = data["n"].copy()
        v_xc_raw = data["v_xc"].copy()
    v_xc = v_xc_raw - v_xc_raw[-1]
    return x, n, v_xc


def load_inner(data_dir, a_left, d):
    x, n, v_xc = load_full(data_dir, a_left, d)
    sl = slice(trim_margin(len(x)), -trim_margin(len(x)))
    return x[sl], n[sl], v_xc[sl]
