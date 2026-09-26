# Status board — icml2027

Updated by the agent after every wave. Newest entry first. Keep every entry; never rewrite
history. Counts come from `python icml2027/scripts/queue.py status`, not from memory.

## Gates

| Gate | Criterion | Status | Evidence |
|---|---|---|---|
| G0 P0 done | `pytest tests/optim` green; orientation regression test passes on the `evoprompt_iter2` artefacts (per-item mean correlation 0.09 → ≥ 0.60); preflight sbatch passes on Euler | open | — |
| G0b Pre-registration frozen | `PREREGISTRATION.md` committed with hash + OSF link | open | — |
| G1 Pilot | On Qwen3.6-35B-A3B × cluster 0, GA and GEPA improve RPS_cal over the base persona with paired-bootstrap CI excluding 0 on the frozen panel, and `m*_interp` rises | open | — |
| G2 Tier-1 complete | ≥ 80% of Tier-1 cells `completed` with 3 seeds; E6 transfers computed | open | — |
| G3 Results frozen | `results/aggregates/` regenerated from scratch; every table reproducible from `results/` | open | — |

## Cell counts

| Wave | Submitted | Completed | Failed (OOM) | Failed (readout) | Stuck | Notes |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | nothing submitted yet |

## Journal

### 2026-09-27 — implementation deployed, waiting for GPUs
- `src/optim/` implemented: calibrated-RPS fitness (alpha cross-fitted in the optimisation panel), orientation
  map, exact isotonic projection, token-exact belief readout through vLLM `/v1/completions` with the model's own
  chat template, budget ledger, eleven arms (GA, DE, PromptBreeder, MAP-Elites, NSGA-II, OPRO, ProTeGi, GEPA,
  random, paraphrase, best-of-B), frozen evaluation with wrong-persona control, cell runner, queue.
- Deviation from plan 3: the mutator (Qwen3.8-27B) runs as ONE long-lived shared service job
  (`slurm/mutator.sbatch`) instead of a second server inside every cell job, so each cell job needs one GPU.
- Deviation: frozen-evaluation calibration centres on the calibration panel's mean belief (as the audit's h26 does).
- `m*` pricing reuses the audit's decoder (`h26_answer_equivalent.fit_decoder`) on the cell's own panels.
- Non-LLM baselines computed (results/baselines). Under RPS the cluster prior (0.124-0.155) beats the cluster-mean
  constant (0.183-0.231) in every cluster, as a proper score must; split-noise SD of the prior is ~0.002.
  Note: the "human" row is a random same-cluster respondent used as a point forecast (0.24-0.31), which a proper
  probabilistic score penalises; it is not a ceiling for probabilistic simulators and must be labelled so.
- Euler: all 8 GPUs were taken by other jobs of this account at submission; icml jobs 7280 (mutator),
  7281/7282 (probes) and 7285 (pilot) queued. The scheduler is strict-priority, so CPU jobs also wait:
  baselines were run locally instead.

### 2026-09-25 — repository organised
- `icml2027/` created with plan, pre-registration draft, agent instructions, configs, Slurm
  templates and results contract. Implementation plans under `docs/superpowers/plans/`.
- Known defects to fix before any GPU job: orientation not handled in
  `src/utils/personality_match.py`; fitness is `S₀`; only `GAEvoluter` exists.
