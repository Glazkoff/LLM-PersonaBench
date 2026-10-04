# Task 2 optimizer adapter and budget report — 2026-10-04

Implemented in `src/optim/arms/*` and `src/optim/budget.py`. This is an engineering fixture report, not a scientific result.

## Behavior

- `BudgetLedger` uses integer executed respondent–item forecasts and a configured full-panel denominator. It rejects an evaluator request before it exceeds `B * N_opt * N_items`. `full_panel_equivalents` is the exact executed ratio; `candidate_evaluations_exact` retains a compatibility alias. `proposal_attempts`, `invalid_proposals`, `mutator_calls`, `hypermutation_calls`, evaluator calls, and token totals are separate counters. A proposal that fails schema parsing consumes an attempt and its mutator tokens but zero forecasts. Invalid proposals are bounded by `max_invalid_proposals` (default `max(20, 5B)`).
- All existing local search arms use this ledger through `Arm._eval`. OpenAI-compatible mutation calls receive an explicit RNG-derived seed; PromptBreeder hypermutation calls are seeded and their tokens appear in history and ledger. The exact genotype returned by every arm is persisted as `selected_genotype.json`, with the matching full-panel candidate ID when available. The GEPA-style arm no longer fills a fractional remainder with a ledger-only charge.
- Registered `gepa_official` separately from local `gepa` (GEPA-style). It requires `gepa==0.1.4` and calls the package's `gepa.optimize` with a real adapter, Pareto selection, one canonical JSON component, an explicit reflection minibatch and seed, no evaluation cache, and the same opt-panel readout. Its higher-is-better respondent scores are negative **raw** RPS. The adapter validates exact genotype keys, exposes only opt-panel diagnostics to reflection, logs actual batch evaluations, and enforces the outer forecast cap before each readout call. GEPA's own metric-call limit remains advisory; the ledger records `gepa_total_metric_calls` for reconciliation. On a refused forecast request, the arm returns its already fully scored elite and marks the native counter unavailable.
- `strict_genotype(text, template)` rejects missing or extra top-level and nested keys, blank nested entries, and non-string scalar or nested values before the legacy parser can fill fields from a template. Campaign arms use it when `arm_cfg.strict_genotype` is true; legacy cells retain the previous parsing behavior.

## Verification

Installed official `gepa==0.1.4` into `/private/tmp/gepa-0.1.4-venv` solely for a local fixture; the shared repository `.venv` was not changed. Scoped command:

`PYTHONPATH=/private/tmp/gepa-0.1.4-venv/lib/python3.12/site-packages UV_CACHE_DIR=/private/tmp/hbs-plan-uv-cache uv run --python .venv/bin/python --no-project python -m pytest tests/optim/test_budget_adapters_v2.py tests/optim/test_arms_and_runner.py -q`

The native fixture checks that `gepa.optimize` evaluates a new valid JSON candidate, selects its known better genotype, logs a full-panel match, and stays below the outer cap. A second fixture checks that invalid native proposals terminate without phantom forecast spending. Local-arm fixtures cover failed parse accounting, PromptBreeder hypermutation tokens, and returned-genotype serialization. These are deterministic test data, not participant evidence.

The complete `tests/optim` suite passed: **56 passed**. Its former fractional-budget fixture now states the denominator and executed passes explicitly and rejects a ledger-only fraction.

## Limits and deployment interface

The campaign environment must pin/install official `gepa==0.1.4`; a registered arm without this dependency raises at runtime. Native GEPA's result is based only on opt-panel raw RPS. The common runner must independently select and evaluate held-out panels after search, and must accept legitimate underuse of `B` (for example, 11.8/12 equivalents in the GEPA-style fixture). `Fitness._beliefs`, `Fitness.Y`, and `Fitness.tgt` are the current common readout/scorer adapter interface. If they change, the native adapter must be updated together with a corresponding fixture. This task did not implement native GigaEvo; that interface is owned by the root task.
