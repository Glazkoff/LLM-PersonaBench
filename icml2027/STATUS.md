# Status board — icml2027

Updated by the agent after every wave. Newest entry first. Keep every entry; never rewrite
history. Counts come from `python icml2027/scripts/queue.py status`, not from memory.

## Gates

| Gate | Criterion | Status | Evidence |
|---|---|---|---|
| G0 P0 done | `pytest tests/optim` green; orientation regression test passes on the `evoprompt_iter2` artefacts (per-item mean correlation 0.09 → ≥ 0.60); preflight sbatch passes on Euler | open | — |
| G0b Pre-registration frozen | `PREREGISTRATION.md` committed with hash + OSF link | open | — |
| G1 Pilot | On Qwen3.6-35B-A3B × cluster 0, GA and GEPA improve RPS_cal over the base persona with paired-bootstrap CI excluding 0 on the frozen panel, and `m*_interp` rises | passed | GA 3/3 seeds CI<0, GEPA CI<0; m* rises (+3.5 GA, +1.9 GEPA mean over completed cells); see RESULTS_BRIEF.md |
| G2 Tier-1 complete | ≥ 80% of Tier-1 cells `completed` with 3 seeds; E6 transfers computed | open | — |
| G3 Results frozen | `results/aggregates/` regenerated from scratch; every table reproducible from `results/` | open | — |

## Cell counts

| Wave | Submitted | Completed | Failed (OOM) | Failed (readout) | Stuck | Notes |
|---|---|---|---|---|---|---|
| tier1 (7388) | 624 cells / 20 groups | 1 | 0 | 0 | 0 | gpt-oss/GigaChat excluded before submission |
| pause 2026-09-28 | 780 cells (tier1+e5+e8) | 89 | 0 | 0 | 0 | 10 failed_other (cache KeyError, fixed; retry on resume) |

## Journal

### 2026-09-29 01:57 MSK — held again after mutator 7906 started
- 7461_13 (granite cluster 0) started before any mutator, waited 1 h and exited 4 ("no mutator service"); its cells stay
  pending for the retry pass. Mutator 7906 took that GPU at 01:55; about two minutes later every pending job of the account
  was held again (336 user holds, 7 admin holds).
- Babysit now keeps a running idle mutator unless another job is actually waiting for a GPU (no job can use it
  while everything is held), so the next release does not queue the mutator from scratch.

### 2026-09-29 00:55 MSK — holds released by the user; campaign running
- The holds were lifted in steps; babysit cancelled mutator 7892 while the cells were still held, and cell 7461_13 then
  started without a mutator. Mutator resubmitted as 7906 (first in the queue). Babysit now cancels a mutator only
  when it is RUNNING with no runnable cells; a pending mutator is left in place.

### 2026-09-28 19:09 MSK — released at 18:50, held again at ~19:05 (mutator 7892 included)
- Babysit changed so that it never cancels or re-nices a job held by the user (mutator or cells); it only acts on
  non-held hbs-icml jobs. Cell counts unchanged (89 completed, 678 pending).

### 2026-09-28 18:44 MSK — account-wide hold again (359 jobs); idle mutator 7852 cancelled
- Mutator 7852 started at 18:42; within two minutes every pending job of the account, hbs-icml cells included, was held.
  Holds left alone; babysit cancelled the idle mutator. Babysit now keeps pending hbs-icml-cells arrays at nice=500
  while the mutator is pending (0 once it runs), so on the next release the mutator starts before our cells.

### 2026-09-28 16:00 MSK — hbs-icml holds released by the user (other account jobs stay held)
- Mutator resubmitted as 7852. Its FIFO priority was below the 7461 cell tasks, which would have started, waited 1 h
  for a mutator and exited; 7461 set to nice=10 (own jobs only) so the mutator starts first. Unheld mage jobs keep
  their place ahead of the campaign.

### 2026-09-28 (afternoon) — queue held again account-wide
- Around 15:50 MSK every pending job of the account (177, including all hbs-icml cell tasks of 7461/7522/7523) was put on
  user hold again. Holds are left alone; the campaign resumes when the user releases them.
- Mutator 7673 had just started (after the user-approved nice=300 on other jobs); with no runnable cells it was idle, so
  babysit cancelled it and the temporary nice values were reset to 0 (hbs_nice_ids.txt -> .done). Babysit resubmits
  the mutator as soon as any cell task is not held.
- Cell counts unchanged: 89 completed, 678 pending, 10 failed_other (cache KeyError, retried later), 3 stale running.

### 2026-09-28 — holds released by the user; campaign resumed
- 7461 (groups 1-16), 7522 (E5), 7523 (E8), 7524 (stage) no longer held; mutator resubmitted as 7673; 7461 throttle 3.
- 7461 group 0 (Qwen3.6-35B-A3B, cluster 3) was cancelled during the pause and is not in the array any more; it is
  picked up by babysit's retry pass once the queue drains, together with the 10 cache-KeyError cells.

### 2026-09-28 — campaign paused by the user (queue-wide hold)
- Around 15:30-16:30 on 2026-09-27 almost every pending job on the Euler account (~200: carl, mage, medevo, mars and
  hbs-icml 7522/7523/7524) was placed on user hold. The user confirmed: keep everything held.
- This session had released 7461 before noticing the hold was account-wide; 7461 was re-held and its one task that had
  started (7461_0) was cancelled, so its cells return to pending. The first Tier-1 array (7388) finished normally.
- The idle shared mutator (7292) was cancelled; babysit restarts it only when a non-held cell array exists.
- State at pause: 89 cells completed, 10 failed with the (fixed) fitness-cache KeyError from pre-fix processes,
  3 stale 'running' (reset on resubmission), 678 pending. Resume = `scontrol release 7461 7522 7523 7524`.

### 2026-09-27 (afternoon) — full pipeline chained on Euler; jobs renamed hbs-
- All campaign jobs renamed with the `hbs-` prefix (live jobs via scontrol, scripts via --job-name).
- Fixed: fitness cache could evict the entry just written when an identical genotype was re-evaluated
  (GEPA KeyError in gepa__Qwen3.6__c1__s3); LRU insert + regression test. Completed cells were unaffected.
- Throughput: prefix-caching models now split each seed's arms over two processes (6 per GPU); Qwen3.6 stays at 3.
- New code: E8 mutator sensitivity (evaluated model as its own mutator, or Qwen3.5-9B served on the cell GPU),
  cross-cluster transfer, m* on IPIP-NEO-300 unseen items, C2ST with style-only rung, pre-registered headline
  selection, H1-H6 tests and RESULTS_BRIEF.md writer, transfer array, stage job, pipeline.sh.
- Chain on Euler: Tier-1 7388 -> 7461 -> E5 7522 (60 cells) -> E8 7523 (96 cells) -> stage 7524
  (headline seeds 4-5 for the winning arm on two models -> transfer array -> CPU analysis). At most 3 cell GPUs
  plus the shared mutator at any time. Babysit resubmits unfinished cells of every grid as one chain once the
  queue is empty.

### 2026-09-27 (early morning) — first result, throughput fixes, GPU contention
- First completed cell (GA, Qwen3.6-35B-A3B, cluster 0, seed 1): calibrated RPS 0.1399 -> 0.1349,
  delta -0.0049 [-0.0062, -0.0037]; below the cluster prior (0.1406) and the wrong-persona control (0.1482);
  VR_between 0.024 -> 0.161; m* 0.42 -> 3.42 answers.
- Readout: prompts now sent as token ids (text prompts lost gpt-oss's prefilled channel); two-phase cache warm-up.
- gpt-oss-20b excluded (0.0 mass on digits under vLLM, both text and token-id prompts); GigaChat3-10B excluded
  (vLLM server fails, MLA). GLM-4.7-Flash and granite-4.2-30b promoted to Tier 1. H1 ablation moved to gemma-4-12B.
- vLLM metrics showed prefill capped at ~15 concurrent prompts (868 waiting, KV cache 2%) and zero prefix-cache hits
  on the hybrid Qwen3.6; --max-num-batched-tokens raised to 65536.
- HSE second site attempted and dropped: the shared account is at its 1 TB quota and no scratch/project space
  exists for proj_1759; the model files this session downloaded there were removed again.
- Restarting the Tier-1 array to apply the throughput fix released 3 GPUs that ~30 queued carl-* jobs took
  immediately; array 7388 (20 groups, %3) now waits behind carl jobs 7383-7386.

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
