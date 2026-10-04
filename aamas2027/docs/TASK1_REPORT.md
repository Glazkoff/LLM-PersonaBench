# Task 1 scoring and fitness report

Implemented in `src/optim/scoring.py`, `src/optim/fitness.py`, and `tests/optim/test_scoring_v2.py`.

- `iso_project` now performs the unbounded isotonic L2 projection on the original values and clips the projected result to `[0, 1]`. The previous clip-first result for `(1.4, .5, .9, 1)` was `(.75, .75, .9, 1)`; the corrected result is `(.9333, .9333, .9333, 1)`.
- RPS and S0 validate finite categorical simplexes and integer targets in `1..K`; NaN targets are masked. CDF conversion and calibration validate their shapes, bounds, and monotonicity. An item with no observed training labels or a calibration fold with no observed labels raises `ValueError`.
- `crossfit_rps_cal` fits alpha and the model CDF center on each fitting half and evaluates on the other half. Its new keyword-only `folds` argument accepts a complete partition; `return_cells=True` adds the held-out cell-loss matrix. The default return remains `(score, (alpha_A, alpha_B))`. `SCORER_VERSION = "v2"` identifies the changed estimator.
- `Fitness` draws one fold partition at construction and uses it for every full candidate. Subset evaluations restrict that fixed partition, with a deterministic split if the subset lies entirely in one half. `per_respondent_scores` now stores actual out-of-fold losses for calibrated RPS. The search scalar and raw/S0 aggregates use respondent means, so differing target missingness does not change respondent weighting.
- The new `rps_raw` objective selects by uncalibrated RPS and supplies raw per-person losses to reflective selection. Existing `rps_cal` and `s0` modes remain. For all modes, `FitnessResult.rps_cal_opt` remains the primary search scalar and `rps_cal_true` remains calibrated RPS; this legacy field name is retained for callers.

Verification:

1. Before implementation: `UV_CACHE_DIR=/private/tmp/hbs-plan-uv-cache uv run --python .venv/bin/python --no-project python -m pytest -q tests/optim/test_scoring_v2.py` → **8 failed**, each exposing an intended regression.
2. After implementation, an added all-missing-fold check failed once and was fixed.
3. Focused final: `UV_CACHE_DIR=/private/tmp/hbs-plan-uv-cache uv run --python .venv/bin/python --no-project python -m pytest -q tests/optim/test_scoring_v2.py tests/optim/test_core.py tests/optim/test_arms_and_runner.py tests/optim/test_checks.py tests/optim/test_queue.py` → **36 passed**.
4. A simultaneous integration addition, `tests/optim/test_campaign_v2.py`, currently blocks collection of the whole `tests/optim` directory because `aamas2027.pipeline` is not yet present. This is outside Task 1 ownership and was reported to the root agent.
5. `git diff --check` → clean.

Limits: This task changes scorer semantics, so historical scores from the prior estimator are not directly comparable. It does not change external frozen-evaluation code or establish any scientific performance claim. A wholly missing respondent contributes NaN to per-person scores and is excluded by the aggregate; coverage should be reported by the analysis layer.
