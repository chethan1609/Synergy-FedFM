# Results Summary

- **Oracle Baseline (F1):** 0.9257

**Pareto-optimal configuration (Φ*) — Sample 0**

- **CCR_mean:** 1180.8657771992623 (≈ 1180.9x)
- **ΔF1_mean:** 0.0340175432107509
- **ΔF1_std:** 0.009669960468588165

**Optimal Hyperparameters (Sample 0)**

- `quantize_bits`: 8
- `M_max`: 989
- `r_k`: 10
- `gamma`: 0.03044891439843533
- `lambda_reg`: 0.04228280551720559

**Notes**

- These values were obtained from `results/lhs_summary.csv` (mean over repeats). The configuration achieves very high compression (≈1180x) with only a 0.034 absolute drop in F1 compared to the Oracle.
- An alternate tradeoff is Sample 10: `CCR_mean=1264.3904297328688`, `ΔF1_mean=0.045462218179663706` (higher compression, slightly larger ΔF1).

_Last updated: automated summary generated during LHS sweep._
