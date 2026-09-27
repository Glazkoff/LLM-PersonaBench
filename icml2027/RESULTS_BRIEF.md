# Results brief — icml2027 (generated 2026-09-27 by scripts/analysis/hypotheses.py)

Every number below is computed from files under `icml2027/results/`. Pending means the inputs do not exist yet.

Completed cells: 83 (Tier-1 main-fitness: 74).

## Tier-1 by arm (mean over completed cells)

| arm | cells | Δ calibrated RPS | cells with CI < 0 | Δ m* (answers) | evolved vs prior | evolved vs wrong-persona |
|---|---|---|---|---|---|---|
| bestofb | 9 | -0.0037 [-0.0043, -0.0032] | 9/9 | +2.99 | -0.0047 | -0.0098 |
| de | 9 | -0.0029 [-0.0037, -0.0020] | 9/9 | +2.23 | -0.0039 | -0.0081 |
| ga | 9 | -0.0042 [-0.0048, -0.0037] | 9/9 | +3.48 | -0.0052 | -0.0113 |
| gepa | 8 | -0.0027 [-0.0037, -0.0016] | 7/8 | +1.91 | -0.0037 | -0.0086 |
| mapelites | 9 | -0.0044 [-0.0049, -0.0039] | 9/9 | +3.63 | -0.0054 | -0.0116 |
| opro | 9 | -0.0036 [-0.0044, -0.0029] | 9/9 | +2.83 | -0.0046 | -0.0101 |
| paraphrase | 9 | -0.0004 [-0.0006, -0.0001] | 5/9 | +0.27 | -0.0014 | -0.0031 |
| promptbreeder | 9 | -0.0040 [-0.0045, -0.0034] | 9/9 | +3.34 | -0.0050 | -0.0100 |
| random | 3 | -0.0048 [-0.0050, -0.0046] | 3/3 | +3.45 | -0.0053 | -0.0109 |

## Pre-registered hypotheses

- **H1 (improper S0 fitness vs calibrated-RPS fitness): pending.** pairs_holding = 0; rule = >= 6 of 8 (model, cluster) pairs
- **H2 (selection-pressure arms beat random and paraphrase): supported.** rule = >= 50% of an arm's cells significantly better (BH q<0.05) than random AND paraphrase
- **H3 (reflective arms reach threshold with fewer evaluations): pending.** n_reflective = 2; n_evolutionary = 13
- **H4 (optimised personas raise m* on unseen IPIP-300 items): pending.** reason = IPIP-NEO-300 transfer not run yet
- **H5 (multi-objective search yields a non-degenerate front): pending.** 
- **H6 (cross-model transfer weaker than cross-cluster): pending.** 
