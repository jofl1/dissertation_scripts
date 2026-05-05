# Dissertation scripts

Scripts to generate plots and data pertaining to the dissertation.

## Section 3

```
section3/
├── make_baseline.py
├── solve_scf.py
├── fig_construction_S04.py
├── fig_matched_pair_S04.py
├── fig_splitting_metric.py
├── fig_krr_mae.py
├── fig_real_space_predictions_S04.py
├── fig_relaxed_density_overlay.py
├── fig_relaxed_response_metrics.py
├── run_relaxed_krr.py
└── utils/
```

Dependencies: `iDEA`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`.

Run order (from inside `section3/`):

```
python make_baseline.py     # regenerates data/baseline_symmetric_uu_d10.npz
python solve_scf.py
python fig_*.py             # any order
python run_relaxed_krr.py
```

Outputs land in `figures/` (PDF + PNG) and `results/` (CSV + NPZ).

## Section 4

```
section4/
├── make_baseline.py
├── solve_asymmetry_scan_d10.py
├── solve_distance_scan.py
├── solve_distance_scan_dA1.py
├── solve_cross_grid.py
├── eigenvalue_table.py
├── fig_5_1_real_space_exact.py
├── fig_5_2_exact_vs_lda_descriptor.py
├── fig_5_3a_asymmetry_scatter.py
├── fig_5_3b_splitting_vs_asymmetry.py
├── fig_5_4a_distance_scatter.py
├── fig_5_4b_splitting_vs_distance.py
├── fig_5_5_master_plot_expanded.py
├── fig_5_6_window_scan_exact.py
├── fig_5_sigma_spatial.py
└── utils/
```

Dependencies: `iDEA`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`.

Run order (from inside `section4/`):

```
python make_baseline.py             # representative AL=1.0, AR=2.0, d=10 system
python solve_asymmetry_scan_d10.py  # the four solves populate the rest of the
python solve_distance_scan.py       # 19-system grid (~30 min total -- 18 KS
python solve_distance_scan_dA1.py   # inversions at tol=1e-5)
python solve_cross_grid.py
python eigenvalue_table.py          # prints tab:eigenvalues to stdout
python fig_*.py                     # any order
```

Outputs land in `figures/` (PDF + PNG); npz solves cache in `data/`.
