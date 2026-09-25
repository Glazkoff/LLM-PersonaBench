# AGENT.md — operating instructions for the autonomous coding agent

You are implementing and running the experiments in `icml2027/PLAN.md` on the Euler cluster,
autonomously, over several weeks. Your objective is the **strongest defensible result under
the pre-registered protocol**, not the largest number. "State of the art" in this paper means:
at equal budget, the winning optimizer beats every baseline (constant floor, random search,
paraphrase search, best-of-B, the hand-written base prompt) and every other optimizer on
calibrated RPS and on `m*`, replicated across seeds and across disjoint respondent panels, with
the human ceiling and the wrong-persona control printed next to it. A result that cannot
survive a hostile reviewer running `cluster_mean` is not a result.

Read, in order: `icml2027/README.md`, `icml2027/PLAN.md` §0 and §4, `icml2027/PREREGISTRATION.md`,
`icml2027/results/README.md`, then the three plans in `docs/superpowers/plans/`. Execute the plans
task by task with `superpowers:executing-plans` (or one subagent per task). Never start a GPU
job before Gate G0 in `icml2027/STATUS.md` is green.

---

## 1. Environment

| Item | Value |
|---|---|
| Cluster | Euler-5, `ssh airi-h200` (135.106.168.26, user `glazkov`, key `~/.ssh/id_ed25519`). 8× NVIDIA H200 NVL 143 GB, 768 CPU cores, 3 TB RAM. Slurm partitions `infer` (GPU) and `train`. |
| Repo on Euler | `/home/glazkov/personality-twins-arr/LLM-PersonaBench` (git; `origin` = `github.com/Glazkoff/LLM-PersonaBench`) |
| Python env | `/home/glazkov/personality-twins-arr/vllmenv/bin/python` (vLLM 0.28.0, torch, transformers). `uv` at `/home/glazkov/.local/bin/uv`. |
| Model cache | `HF_HOME=/home/glazkov/hf_cache`. Cached: Qwen3.6-35B-A3B(-FP8), Qwen3.8-27B, gemma-4-12B-it, gpt-oss-20b, GLM-4.7-Flash, granite-4.2-8b/30b, Apertus-8B, Qwen3-235B-A22B-2507-FP8, GigaChat3-10B. Check with `ls $HF_HOME/hub`. |
| Logs | `/home/glazkov/personality-twins-arr/logs/` (Slurm stdout/err, vLLM logs) |
| Data | `data/raw/df_ipipneo_120_clusters` (410,168 × 167 CSV; columns `case, sex, age, country, i1..i120, <facet/trait scores>, clusters`), `data/IPIP-NEO/120/item_key.csv` (item, sign, reverse, facet_key, facet, text), `data/PAlign/Test-set.json` (300 respondents, IPIP-NEO-300), `data/PAlign/Dark-Triad.csv`, `data/openpsychometrics/{BIG5,HEXACO}`. The corpus is **not** in git; it is on Euler already. |
| Other users | MARS jobs run on `infer`. Use at most **4 GPUs concurrently** unless `squeue` shows the rest idle for > 1 h; never request more than 2 GPUs for one job unless `models.yaml` says `tp: 4`. |
| Second cluster | HSE cHARISMa (`ssh hse`, account `proj_1759`, see `../HSE_RUN.md`) is overflow only; do not pass `--mem` there. |

Nothing runs bare on the login node. Tests run in a CPU Slurm job (`icml2027/slurm/preflight.sbatch`).

## 2. Ground truth you must not rediscover the hard way

1. **Orientation.** The corpus stores human answers reverse-recoded on 55 of 120 items; the
   model answers the literal item. Every comparison flips the model's simplex on those items
   (`src/optim/orientation.py`). The previous team reintroduced this bug three times in three
   scripts. The regression test in `tests/optim/test_orientation.py` is a hard gate.
2. **The audited metric is improper.** Answer similarity `S₀ = 1 − |x−y|/4` is maximised by a
   constant; a real human loses to the cluster mean under it. Fitness is calibrated RPS.
   `S₀` exists only as the H1 ablation arm.
3. **Readout.** Hybrid-thinking models emit `<think>` first; disable thinking and prefill the
   assistant turn (`"My answer is "`; for harmony-format models such as gpt-oss:
   `"<|channel|>final<|message|>My answer is "`). Reading through a chat-completions server
   returned near-zero mass on the answer tokens for some models where the HF forward pass gave
   0.99; the guard `mass_on_scale ≥ 0.5` (mean over the panel) decides per model in preflight.
   If vLLM fails the guard for a model, use the HF forward-pass path
   (`arr2026/scripts/hyp/h21_reachability.py` shows it) for that model and record `readout:
   belief_hf` in `run_meta.json`.
4. **FP8 needs Hopper.** FP8 checkpoints run on H200 only. bf16 everywhere else.
5. **Multi-GPU FP8 produced garbage once.** Run each model on the smallest TP that fits; the
   readout must be validated per (model, TP) in preflight.
6. **The evaluation panel is sacred.** It is touched exactly once per cell by
   `src/optim/frozen_eval.py`. If any search-time code reads it, the cell is invalid.

## 3. Workflow and gates

| Phase | Deliverable | Gate to pass before moving on |
|---|---|---|
| P0 (plan 1) | `src/optim/` package: orientation, scoring+calibration, panels, persona, readout, fitness, budget, results writer; `tests/optim` green | **G0**: `pytest tests/optim -q` passes; orientation regression on `results_experiments/evoprompt_iter2/*` artefacts moves per-item mean correlation from ≈0.09 to ≥0.60; preflight sbatch passes on Euler for every Tier-1 model (mass ≥ 0.5) |
| Freeze | `PREREGISTRATION.md` committed; hash + OSF link written into it | **G0b** |
| E0 pilot (plan 2 partial + plan 3 runner) | GA and GEPA on Qwen3.6-35B-A3B × cluster 0 × seeds {1,2,3} | **G1**: evolved beats base on RPS_cal (paired CI excludes 0) and `m*_interp` rises. If not: stop, write the finding into `STATUS.md`, and switch the paper framing to the H4-false branch per PLAN.md §3.3. Do not tune the protocol to pass the gate. |
| Tier 1 (plan 3) | 7 arms × 6 models × 4 clusters × 3 seeds | **G2**: ≥ 80% cells completed; E6 transfers computed |
| Tier 2 / E5 / E8 | extension models; multi-objective; mutator sensitivity | wave reports in `STATUS.md` |
| Analysis | `results/aggregates/` from `scripts/analysis/`; figures 1–4 | **G3**: `bash icml2027/scripts/analysis/regenerate_all.sh` reproduces every aggregate from scratch |

At every gate: update `STATUS.md`, commit, push, and run one internal review (ARS
`academic-paper-reviewer quick` on the results summary; Codex only with the user's model policy:
`gpt-6-sol` at low or medium reasoning, never xhigh — if the account rejects the model, ask
the user rather than substituting).

## 4. Running jobs on Euler

**Submit** (from the repo root on Euler):
```bash
sbatch icml2027/slurm/preflight.sbatch                 # always first after a code change
python icml2027/scripts/queue.py submit --manifest icml2027/configs/grid_tier1.yaml --max-concurrent 4
python icml2027/scripts/queue.py status                 # per-model counts; also writes icml2027/queue/state.json
```
`queue.py submit` creates every cell directory with `status.json = pending` first, then submits
one Slurm array per model (`cell_group.sbatch`, one task = one cluster, running every (arm,
seed) sequentially against one vLLM server), with `--array=0-3%<concurrency>` and
`--dependency=afterok:<preflight_job>`. It is idempotent: cells whose `status.json` is
`completed` are skipped; cells `failed_oom` with `attempt < 3` are resubmitted with
`--gpu-memory-utilization` lowered by 0.05; `failed_readout` is never retried; anything else
is `stuck` and needs you.

**Job hygiene**
- One vLLM server per job, on `127.0.0.1:$((8000 + RANDOM % 1000))`, killed in the `EXIT` trap.
- Health-check the server for up to 30 min; if it never comes up, exit non-zero and let the
  queue mark `failed_other`.
- Propagate the Python exit code (`rc=$?; exit $rc`) — an `echo` after the run once masked 16
  failed cells as `COMPLETED`.
- Write `status.json` transitions from Python, not from the shell.
- `--time` is 12 h for a cell group; if a group needs more, split by seed, do not raise it.
- Never `scancel` another user's job. Never write outside the repo, `$HF_HOME` and `logs/`.

**Resume after a crash**: `queue.py submit` again with the same manifest. Do not delete
`state.json`; do not delete cell directories.

**Throughput expectation** (verify in E0, record in `STATUS.md`): one candidate = 40
respondents × 60 items = 2,400 single-token forward passes with the persona prefix cached; a
cell = 80 candidates = 192,000 passes; a 30B-class model on one H200 should finish a cell in
5–15 minutes plus server start-up. If a cell takes > 1 h, stop and profile before continuing.

## 5. Results and git discipline

- Results go only under `icml2027/results/` per `results/README.md`. Atomic writes. Nothing deleted.
- Work on branch **`icml2027`**. Commit after every task with tests passing; push at least daily.
  At each gate, fast-forward merge into `main` and push (the user asked for `main` to carry
  the state). End commit messages with the co-author attribution line your harness specifies
  for the model you are running as.
- Never commit: `*.npz`, `*.npy`, `logs/`, `.env`, `data/raw/*`, model weights, API keys.
  `git status` must be clean of those before every commit; `.gitignore` already covers them.
- Every wave: append a row to the cell-count table in `STATUS.md` from `queue.py status`
  output and a journal entry (what ran, what failed, what you changed, with commit hashes).
- Large tensors (`belief_probs.npz`) are collected at G3 into a bundle with a SHA-256 manifest,
  following `arr2026/BUNDLE_REGENERATE.md`.

## 6. Integrity rails (non-negotiable)

1. Fitness = RPS_cal on the optimisation panel; `α` by 2-fold cross-fitting inside that panel
   during search; `α` on the calibration panel at frozen evaluation. Never on the evaluation panel.
2. Budget B = 80 candidate evaluations for every arm. A candidate that fails to parse is
   charged and scored as worst-so-far, never skipped.
3. Seeds, panels and item splits come from `configs/panels.yaml` and are logged in `run_meta.json`.
   No re-drawing a panel because a result looked bad.
4. The pre-registered hypotheses are tested as written. New analyses are labelled exploratory.
5. A failed cell is reported, not deleted. An excluded cell (mass < 0.5) is listed.
6. Tables are generated by `scripts/analysis/`; if a number in a draft cannot be traced to a
   file under `results/`, remove the number.
7. If you find a bug that changes reported numbers, fix it, re-run the affected cells, and record
   the before/after in `STATUS.md`. Do not quietly overwrite.
8. Do not add a metric, panel size, or budget after seeing results in order to make an
   effect appear. If the protocol is wrong, amend `PREREGISTRATION.md` with a dated reason first.

## 7. Stop and ask a human when

- Gate G1 fails (the framing decision is the user's).
- A Tier-1 model fails the readout guard on both vLLM and HF paths (drop or replace is the user's call).
- Projected GPU-hours for a wave exceed 300, or API spend for Tier 3 exceeds the user's budget line.
- Any cluster policy or account issue (`sbatch` refused, quota, other users' complaints).
- A data-licence question (Twin-2K-500 terms, PAlign redistribution) before using a dataset in a figure.
- You are tempted to change the protocol after seeing results.

Otherwise, proceed. Reversible engineering decisions (file layout inside `src/optim/`, test
structure, refactors) are yours.

## 8. Definition of done for the experimental campaign

- `STATUS.md` shows G0–G3 green with evidence links.
- `results/aggregates/` contains: `tier1_summary.csv` (arm × metric, mean over models with CI,
  wins vs floor / random / base), `tier1_cells.csv` (every cell), `mstar_by_arm_model.csv`,
  `transfer_ipip300.csv`, `crossmodel_matrix.csv`, `pareto_fronts.json`,
  `evaluations_to_threshold.csv`, `baselines_splitnoise.csv`, and `figures/fig{1..4}.pdf`.
- `bash icml2027/scripts/analysis/regenerate_all.sh` runs clean on a fresh clone plus the bundle.
- A one-page `RESULTS_BRIEF.md` in `icml2027/` stating, per hypothesis, supported / not
  supported / inconclusive with the number and interval, ready to paste into the paper.
