# icml2027 — Paper B: prompt optimizers for persona simulation under a proper score

This directory holds everything for the second paper that grows out of the withdrawn
NeurIPS 2026 submission #22622 ("Evolutionary Prompt Optimization for Personality-Driven
Digital Twins"). The first paper, the audit ("What Is a Persona Worth?"), lives in
`../arr2026/` and `../paper/`. This one asks the question the audit leaves open:

> Which automatic prompt optimizers, on which models, increase the person-specific
> information an LLM persona carries, measured under a strictly proper score and priced in
> the respondent's own answers (`m*`)?

Primary target: **ICML 2027** (abstract 16 Jan 2027, paper 22 Jan 2027). Fallbacks: ACL 2027
via ARR January 2027, IJCAI 2027 (~1 Feb 2027), NeurIPS 2027 (21 May 2027). The venue is a
label on this directory, not a constraint on its content.

## Read in this order

| File | What it is |
|---|---|
| [`PLAN.md`](PLAN.md) | The resubmission plan: reviewer ledger, reframe, experiment program E0–E10, venues, timeline, risks. The *why*. |
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | Hypotheses H1–H6 with every analysis choice fixed (budget, panels, seeds, tests). Freeze it (commit hash + OSF) before the Tier-1 grid runs. |
| [`AGENT.md`](AGENT.md) | Operating instructions for the autonomous coding agent that implements the code and runs the experiments on Euler. The *how*, with integrity rails and stop conditions. |
| [`../docs/superpowers/plans/2026-09-25-icml2027-p0-proper-score-pipeline.md`](../docs/superpowers/plans/2026-09-25-icml2027-p0-proper-score-pipeline.md) | Implementation plan 1: orientation fix, proper-score fitness, belief readout, panels, budget accounting, results writer. Gating. |
| [`../docs/superpowers/plans/2026-09-25-icml2027-optimizer-arms.md`](../docs/superpowers/plans/2026-09-25-icml2027-optimizer-arms.md) | Implementation plan 2: the optimizer arms (GA, DE, MAP-Elites/GigaEvo, GEPA, OPRO, PromptBreeder, ProTeGi) and the budget-matched baselines. |
| [`../docs/superpowers/plans/2026-09-25-icml2027-euler-runner-and-analysis.md`](../docs/superpowers/plans/2026-09-25-icml2027-euler-runner-and-analysis.md) | Implementation plan 3: Slurm runner with state and retries, frozen evaluation, `m*` pricing, statistics, figures. |
| [`STATUS.md`](STATUS.md) | Live status board: gates, cell counts, blockers. The agent updates it after every wave. |
| [`results/README.md`](results/README.md) | The results contract: directory layout, file schemas, what is tracked in git and what is not. |

## Layout

```
icml2027/
  PLAN.md  PREREGISTRATION.md  AGENT.md  STATUS.md  README.md
  configs/
    models.yaml            model slate: HF id, slug, TP, dtype, readout, release date, tier
    panels.yaml            respondent panel sizes and seeds; item split rule
    grid_tier1.yaml        the Tier-1 run grid (arms x models x clusters x seeds)
    optimizers/*.yaml      per-arm hyperparameters, all at budget B = 80 candidate evaluations
  slurm/
    preflight.sbatch       CPU: tests + orientation regression + readout guard per model
    cell_group.sbatch      GPU: serve one model, run every (arm, seed) for one cluster
    cpu_analysis.sbatch    CPU: baselines, aggregation, bootstrap, m*, figures
  scripts/
    queue.py               manifest -> sbatch arrays with dependencies; `status` reads state
    analysis/              aggregate.py, bootstrap.py, price_mstar.py, figures.py
  queue/                   manifest.json + state.json for the current run (tracked)
  results/
    cells/<cell_id>/       one directory per (arm, model, cluster, seed): see results/README.md
    baselines/             non-LLM floors and ceilings per (model-independent) cluster panel
    aggregates/            tables the paper prints, generated only by scripts/analysis
```

Code lives in the package `src/optim/` (new) and reuses `src/models/providers/local_vllm.py`,
`src/evolution/utils.py` (genotype parsing) and `arr2026/scripts/hyp/{_scoring,_orientation,
_corpora}.py` (scoring rules, orientation, corpora). Tests live in `tests/optim/`.

## Running

Everything runs on Euler (`ssh airi-h200`) through Slurm; nothing runs bare on the login node.

```bash
# on Euler, from /home/glazkov/personality-twins-arr/LLM-PersonaBench
sbatch icml2027/slurm/preflight.sbatch                       # tests + readout guard
python icml2027/scripts/queue.py submit --manifest icml2027/configs/grid_tier1.yaml --max-concurrent 4
python icml2027/scripts/queue.py status                       # counts by state, per model
sbatch icml2027/slurm/cpu_analysis.sbatch                     # after a wave completes
```

## Conventions

- Model slug = HF id with `/` → `_` and `.` → `p` (matches `arr2026/slurm/euler/*.sbatch`).
- Cell id = `{arm}__{model_slug}__c{cluster}__s{seed}`.
- Every number in the paper is produced by a script under `scripts/analysis/` from files under
  `results/`; nothing is typed by hand into a table.
- Orientation: model answers are raw, the corpus is recoded on 55 of 120 items. One transform
  (`src/optim/orientation.py`) is imported by every scorer. A regression test guards it.
- Selection happens on the optimisation panel, hyperparameters (`α`, ridge) on the calibration
  panel, every reported number on the frozen evaluation panel. No exceptions.
