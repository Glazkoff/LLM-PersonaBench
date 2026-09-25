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

### 2026-09-25 — repository organised
- `icml2027/` created with plan, pre-registration draft, agent instructions, configs, Slurm
  templates and results contract. Implementation plans under `docs/superpowers/plans/`.
- Known defects to fix before any GPU job: orientation not handled in
  `src/utils/personality_match.py`; fitness is `S₀`; only `GAEvoluter` exists.
