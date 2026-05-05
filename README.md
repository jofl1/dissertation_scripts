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

## Appendices

```
appendices/
├── appendix_B_window_control.py    # tab:window_control (per-ell vs common grid)
├── appendix_C_gradient_descriptors.py  # tab:gradient_descriptors_exact (D1/D2/D3 across 19 systems)
└── appendix_D_sigma_k_robustness.py    # rho(k) sweep over k in {3,5,7,10,15,20}
```

Each script reads from `../section4/data/`, so the section 4 solves must
have finished before the appendices can run. Outputs land in
`appendices/results/`.

Run order (from inside `appendices/`):

```
python appendix_B_window_control.py     # ~3 min
python appendix_C_gradient_descriptors.py  # ~10 min (4 descriptors × 19 systems)
python appendix_D_sigma_k_robustness.py    # ~7 min  (KRR once per system + KNN sweep)
```

Appendix A in the dissertation is `fig_5_sigma_spatial.pdf`, generated
by `section4/fig_5_sigma_spatial.py`.
