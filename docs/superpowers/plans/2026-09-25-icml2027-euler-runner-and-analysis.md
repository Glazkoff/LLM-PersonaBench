# Euler Runner and Analysis Implementation Plan (icml2027, plan 3 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the grid on Euler unattended (cell runner, frozen evaluation, Slurm queue with state and retries) and turn `icml2027/results/` into every table and figure the paper prints (baselines, bootstrap + BH, `m*` pricing, figures).

**Architecture:** `src/optim/run_cell.py` executes all cells of one (model, cluster) against one vLLM server: build panels → `Fitness` → arm → `frozen_eval` (base and evolved, once) → `status.json`. `icml2027/scripts/queue.py` expands a grid YAML into cells, creates their directories, submits one Slurm array per model, and reads status files back. Analysis scripts are pure functions of `results/`.

**Tech Stack:** Python 3.11, numpy, pandas, PyYAML, matplotlib, Slurm (`sbatch`, `squeue`, `sacct`), scikit-learn (C2ST). Consumes plans 1 and 2.

**Spec:** `icml2027/PLAN.md` §4 E3, E4, E6, E7, E8; `icml2027/PREREGISTRATION.md` §3–6; `icml2027/results/README.md`; `icml2027/AGENT.md` §4.

## Global Constraints

- The evaluation panel is read only by `src/optim/frozen_eval.py`, once per cell; `write_eval` refuses a second write.
- α and ridge penalties at frozen evaluation are selected on the calibration panel.
- Bootstrap B = 2000, seed 20260925, percentile 95% intervals, paired over evaluation respondents; BH q < 0.05 across the full primary-test family.
- A cell's Slurm exit code must reflect its Python exit code. Status transitions are written by Python.
- Queue concurrency default 4 GPUs; never more than `max_concurrent_arrays` arrays in flight.
- `m*` reuses the audit protocol (`arr2026/scripts/hyp/h26_answer_equivalent.py`): budgets {0,1,2,3,5,8,12,20,30,45,60}, ridge retuned per budget on the calibration panel, interpolated crossing reported.

---

### Task 1: Frozen evaluation

**Files:**
- Create: `src/optim/frozen_eval.py`, `tests/optim/test_frozen_eval.py`

**Interfaces:**
- Consumes: `Panels`, `scoring.{cdf, pop_cdf, calibrate, cdf_to_simplex, rps, s0, select_alpha}`, `orientation.flip_simplex`, `persona.system_prompt`, `panels.{ITEM_SPLIT, answers, persona_scores}`, `readout.beliefs`, `CellWriter.write_eval`.
- Produces: `evaluate_frozen(panels, readout, base_g, evolved_g, alphas, boot_B=2000, boot_seed=20260925) -> dict` matching `eval_frozen.json` in `results/README.md` (without `mstar`, filled in Task 5) and returning also `per_respondent` arrays; `paired_boot_ci(a, b, B, seed) -> (mean, lo, hi)`; `floors(panels) -> dict` (`cluster_mean_rps` = point mass at the rounded per-item train mean; `cluster_prior_rps` = `F0` as a forecast; `human_rps` = a random train respondent's answers as a point mass, averaged over 20 draws); `wrong_persona(panels, readout, g, alphas, rng) -> float` (evolved genotype with personas permuted within the eval panel).

- [ ] **Step 1: Write the failing test**

```python
# tests/optim/test_frozen_eval.py
import numpy as np, pandas as pd
from src.optim.frozen_eval import evaluate_frozen, paired_boot_ci, floors
from src.optim.panels import make_panels
from src.optim.readout import FakeReadout
from src.optim.genotype import seed_genotype

def _panels():
    rng = np.random.default_rng(0); n = 900
    df = pd.DataFrame(rng.integers(1, 6, size=(n, 120)).astype(float), columns=[f"i{i}" for i in range(1, 121)])
    df["case"] = np.arange(n); df["clusters"] = 0
    return make_panels(df, 0, 1, {"train": 500, "opt": 40, "cal": 40, "eval": 100})

def test_paired_boot_ci_detects_shift():
    rng = np.random.default_rng(0); a = rng.normal(0, 1, 200); b = a + 0.5
    m, lo, hi = paired_boot_ci(a, b, 500, 1); assert lo > 0.4 and hi < 0.6

def test_evaluate_frozen_schema():
    p = _panels(); g = seed_genotype(0)
    out = evaluate_frozen(p, FakeReadout(np.random.default_rng(1)), g, g, [0, .5, 1], boot_B=200)
    for k in ("panel", "base", "evolved", "delta", "floors"): assert k in out
    assert abs(out["delta"]["rps_cal"]) < 0.05 and len(out["per_respondent"]["base_rps_cal"]) == 100

def test_floors_order_on_random_data():
    f = floors(_panels()); assert set(f) == {"cluster_mean_rps", "cluster_prior_rps", "human_rps"}
```

- [ ] **Step 2: Run** → FAIL

- [ ] **Step 3: Implement**

```python
# src/optim/frozen_eval.py
import hashlib
import numpy as np
from src.optim import scoring as S
from src.optim.orientation import flip_simplex
from src.optim.panels import ITEM_SPLIT, answers, persona_scores
from src.optim.persona import system_prompt

def paired_boot_ci(a, b, B=2000, seed=20260925):
    d = np.asarray(b) - np.asarray(a); rng = np.random.default_rng(seed); n = len(d)
    bs = np.array([np.nanmean(d[rng.integers(0, n, n)]) for _ in range(B)])
    return float(np.nanmean(d)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def _beliefs(readout, g, df, tgt):
    inp = ITEM_SPLIT[0]
    systems = [system_prompt(g, persona_scores(x)) for x in answers(df, inp)]
    P, M, _ = readout.beliefs(systems, tgt); return flip_simplex(P, tgt), M

def _score_arm(readout, g, panels, F0, alphas):
    tgt = ITEM_SPLIT[1]; Ycal, Yev = answers(panels.cal, tgt), answers(panels.eval, tgt)
    Pc, _ = _beliefs(readout, g, panels.cal, tgt); a, _ = S.select_alpha(S.cdf(Pc), F0, Ycal, alphas)
    Pe, Me = _beliefs(readout, g, panels.eval, tgt)
    Pcal = S.cdf_to_simplex(S.calibrate(S.cdf(Pe), F0, a))
    per = S.per_respondent(S.rps(Pcal, Yev))
    mu = (Pe * np.arange(1, 6)).sum(-1)
    vr = float(np.var(mu, 0).mean() / np.nanvar(Yev, 0).mean())
    return {"rps_cal": float(np.nanmean(per)), "rps_raw": float(np.nanmean(S.rps(Pe, Yev))), "s0": float(np.nanmean(S.s0(Pe, Yev))),
            "alpha": a, "vr_between": vr, "mass_on_scale": float(Me.mean())}, per, Pe

def floors(panels, draws=20, seed=0):
    tgt = ITEM_SPLIT[1]; Ytr, Yev = answers(panels.train, tgt), answers(panels.eval, tgt); n, J = Yev.shape
    def point(v):
        P = np.zeros((n, J, 5)); P[np.arange(n)[:, None], np.arange(J)[None, :], (np.clip(v, 1, 5) - 1).astype(int)] = 1; return P
    cm = np.broadcast_to(np.rint(np.nanmean(Ytr, 0)), (n, J))
    F0 = S.pop_cdf(Ytr); rng = np.random.default_rng(seed)
    human = np.mean([np.nanmean(S.rps(point(np.broadcast_to(Ytr[rng.integers(len(Ytr))], (n, J))), Yev)) for _ in range(draws)])
    return {"cluster_mean_rps": float(np.nanmean(S.rps(point(cm), Yev))),
            "cluster_prior_rps": float(np.nanmean(S.rps(S.cdf_to_simplex(np.broadcast_to(F0, (n, J, 4))), Yev))),
            "human_rps": float(human)}

def wrong_persona(panels, readout, g, alphas, F0, rng):
    tgt = ITEM_SPLIT[1]; Yev = answers(panels.eval, tgt)
    shuffled = panels.eval.iloc[rng.permutation(len(panels.eval))]
    Pc, _ = _beliefs(readout, g, panels.cal, tgt); a, _ = S.select_alpha(S.cdf(Pc), F0, answers(panels.cal, tgt), alphas)
    Pe, _ = _beliefs(readout, g, shuffled, tgt)
    return float(np.nanmean(S.rps(S.cdf_to_simplex(S.calibrate(S.cdf(Pe), F0, a)), Yev)))

def evaluate_frozen(panels, readout, base_g, evolved_g, alphas, boot_B=2000, boot_seed=20260925, wrong_persona_seed=0):
    tgt = ITEM_SPLIT[1]; F0 = S.pop_cdf(answers(panels.train, tgt))
    base, pb, _ = _score_arm(readout, base_g, panels, F0, alphas)
    evo, pe, Pe = _score_arm(readout, evolved_g, panels, F0, alphas)
    m, lo, hi = paired_boot_ci(pb, pe, boot_B, boot_seed)
    ids = ",".join(map(str, sorted(panels.eval["case"].tolist())))
    evo["wrong_persona_rps_cal"] = wrong_persona(panels, readout, evolved_g, alphas, F0, np.random.default_rng(wrong_persona_seed))
    return {"panel": {"cluster": panels.cluster, "seed": panels.seed, "n_eval": len(panels.eval),
                      "respondent_ids_sha256": hashlib.sha256(ids.encode()).hexdigest()},
            "base": base, "evolved": evo,
            "delta": {"rps_cal": m, "rps_cal_ci95": [lo, hi], "boot_B": boot_B, "boot_seed": boot_seed},
            "floors": floors(panels), "mstar": None, "transfer_ipip300": None,
            "excluded": evo["mass_on_scale"] < 0.5, "exclusion_reason": "mass_on_scale<0.5" if evo["mass_on_scale"] < 0.5 else None,
            "per_respondent": {"base_rps_cal": pb.tolist(), "evolved_rps_cal": pe.tolist()}}
```

`per_respondent` is popped by the caller and written to `per_respondent.json`; `Pe` (evolved eval beliefs) is saved to `belief_probs.npz` by the caller for `m*` and C2ST.

- [ ] **Step 4: Run** → 3 passed. **Step 5: Commit**

```bash
git add src/optim/frozen_eval.py tests/optim/test_frozen_eval.py
git commit -m "optim: frozen evaluation with calibration-panel alpha, floors, wrong-persona control, paired bootstrap"
```

---

### Task 2: Cell runner

**Files:**
- Create: `src/optim/run_cell.py`, `tests/optim/test_run_cell.py`

**Interfaces:**
- Consumes: `ARMS`, `LLMMutator`, `Fitness`, `BudgetLedger`, `CellWriter`, `cell_id`, `make_panels`, `excluded_cases`, `load_corpus`, `seed_genotype`, `VLLMReadout`, `evaluate_frozen`.
- Produces: CLI `python -m src.optim.run_cell --manifest M --model-slug S --cluster C --slurm-job-id J [--workers 48] [--fake]`. Function `run_one(cell: dict, df, readout, mutator, results_root, meta_extra) -> str` returning the final state. Manifest cell dict: `{"cell_id","arm","fitness","model_slug","hf_id","prefill","cluster","seed","budget_B","arm_cfg":{...}}`. Exit code: 0 if every cell ended `completed` or `failed_readout`; 1 otherwise.
- States: `running` → `completed` | `failed_readout` (mass < 0.5 on the seed genotype's first evaluation; no retry) | `failed_oom` (CUDA OOM text in an exception or server gone) | `failed_other`. Cells already `completed` are skipped.

- [ ] **Step 1: Write the failing test** (runs the full cell path with fakes)

```python
# tests/optim/test_run_cell.py
import json, numpy as np, pandas as pd
from src.optim.run_cell import run_one
from src.optim.readout import FakeReadout
from tests.optim.conftest import TEMPLATE

class M:
    def complete(self, s, u, temperature=0.7, max_tokens=1200):
        from src.optim.genotype import seed_genotype
        g = seed_genotype(0); g["role_definition"] += " tweak"; return json.dumps(g), 10, 10

def test_run_one_completes(tmp_path):
    rng = np.random.default_rng(0); n = 1200
    df = pd.DataFrame(rng.integers(1, 6, size=(n, 120)).astype(float), columns=[f"i{i}" for i in range(1, 121)])
    df["case"] = np.arange(n); df["clusters"] = 0
    cell = {"cell_id": "random__fake__c0__s1", "arm": "random", "fitness": "rps_cal", "model_slug": "fake", "hf_id": "fake",
            "cluster": 0, "seed": 1, "budget_B": 4, "arm_cfg": {"samples": 4},
            "panel_sizes": {"train": 500, "opt": 40, "cal": 40, "eval": 100}, "alphas": [0, .5, 1], "exclude_globs": [], "boot_B": 100}
    state = run_one(cell, df, FakeReadout(np.random.default_rng(1)), M(), tmp_path, {"git_sha": "x"})
    d = tmp_path / cell["cell_id"]
    assert state == "completed"
    assert len((d / "history.jsonl").read_text().splitlines()) == 4
    assert json.loads((d / "status.json").read_text())["state"] == "completed"
    assert "delta" in json.loads((d / "eval_frozen.json").read_text())
    assert run_one(cell, df, FakeReadout(np.random.default_rng(1)), M(), tmp_path, {}) == "completed"  # idempotent skip
```

- [ ] **Step 2: Run** → FAIL

- [ ] **Step 3: Implement**

```python
# src/optim/run_cell.py
import argparse, json, os, platform, subprocess, sys, traceback
from pathlib import Path
import numpy as np
from src.optim.arms import ARMS
from src.optim.arms.base import LLMMutator
from src.optim.budget import BudgetLedger
from src.optim.fitness import Fitness
from src.optim.frozen_eval import evaluate_frozen
from src.optim.genotype import seed_genotype
from src.optim.panels import excluded_cases, load_corpus, make_panels
from src.optim.results import CellWriter, atomic_write_json, now

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "icml2027/results/cells"

def _git_sha():
    try: return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception: return "unknown"

def run_one(cell, df, readout, mutator, results_root, meta_extra) -> str:
    w = CellWriter(results_root, cell["cell_id"])
    st = w.dir / "status.json"
    if st.exists() and json.loads(st.read_text())["state"] == "completed": return "completed"
    attempt = (json.loads(st.read_text()).get("attempt", 0) + 1) if st.exists() else 1
    if (w.dir / "history.jsonl").exists(): w.archive_attempt()
    w.set_status("running", attempt=attempt, slurm_job_id=meta_extra.get("slurm_job_id"))
    w.write_config(cell)
    meta = {**meta_extra, **{k: cell[k] for k in ("cell_id", "arm", "fitness", "model_slug", "hf_id", "cluster", "seed", "budget_B")},
            "started": now(), "git_sha": meta_extra.get("git_sha", _git_sha())}
    try:
        panels = make_panels(df, cell["cluster"], cell["seed"], cell["panel_sizes"], excluded_cases(cell.get("exclude_globs", [])))
        rng = np.random.default_rng(cell["seed"] * 7919 + cell["cluster"])
        fit = Fitness(panels, readout, cell["alphas"], rng, objective=cell.get("fitness", "rps_cal"))
        seed_g = seed_genotype(cell["cluster"])
        probe = fit.evaluate(seed_g)                   # not charged: readout guard only, never used for selection
        meta["mass_on_scale_mean"] = probe.mass
        if probe.mass < 0.5:
            meta["readout_guard_passed"] = False; w.write_meta(meta)
            w.set_status("failed_readout", attempt=attempt, message=f"mass {probe.mass:.3f}"); return "failed_readout"
        meta["readout_guard_passed"] = True
        arm = ARMS[cell["arm"]]({**cell["arm_cfg"]}, rng, mutator, seed_g)
        ledger = BudgetLedger(cell["budget_B"])
        best = arm.run(fit, seed_g, ledger, w)
        if ledger.used != cell["budget_B"]: raise RuntimeError(f"budget not exhausted: {ledger.used}/{cell['budget_B']}")
        w.finish(ledger)
        ev = evaluate_frozen(panels, readout, seed_g, best, cell["alphas"], boot_B=cell.get("boot_B", 2000))
        atomic_write_json(w.dir / "per_respondent.json", ev.pop("per_respondent"))
        w.write_eval(ev)
        meta.update(ledger.to_dict()); meta["finished"] = now(); w.write_meta(meta)
        w.set_status("completed", attempt=attempt); return "completed"
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"; tb = traceback.format_exc()[-4000:]
        state = "failed_oom" if ("out of memory" in tb.lower() or "Connection refused" in tb) else "failed_other"
        meta["error"] = tb; w.write_meta(meta); w.set_status(state, attempt=attempt, message=msg[:500]); return state

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--model-slug", required=True)
    ap.add_argument("--cluster", type=int, required=True); ap.add_argument("--slurm-job-id", default="local")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    man = json.load(open(a.manifest)); cells = [c for c in man["cells"] if c["model_slug"] == a.model_slug and c["cluster"] == a.cluster]
    if not cells: print("no cells"); sys.exit(0)
    from src.optim.readout import VLLMReadout
    readout = VLLMReadout(cells[0]["hf_id"], prefill=cells[0].get("prefill", "My answer is "), workers=a.workers)
    mut = man["mutator"]; mutator = LLMMutator(mut["hf_id"], base_url=os.environ.get("MUTATOR_BASE_URL"))
    df = load_corpus(); states = []
    meta = {"slurm_job_id": a.slurm_job_id, "node": platform.node(), "git_sha": _git_sha(), "prereg_sha": man.get("prereg_sha")}
    for c in sorted(cells, key=lambda c: (c["seed"], c["arm"])):
        s = run_one(c, df, readout, mutator, RESULTS, meta); states.append(s); print(f"{c['cell_id']}: {s}", flush=True)
    sys.exit(0 if all(s in ("completed", "failed_readout") for s in states) else 1)

if __name__ == "__main__":
    main()
```

Mutator serving: the mutator (Qwen3.8-27B) must be reachable at `MUTATOR_BASE_URL`. When the evaluated model *is* the mutator model, reuse `LOCAL_LLM_BASE_URL`. Otherwise `cell_group.sbatch` must request 2 GPUs and start a second vLLM server for the mutator on another port; add that to the sbatch in Step 4 below.

- [ ] **Step 4: Extend `icml2027/slurm/cell_group.sbatch` for the mutator**

Add after the evaluated-model health check (and change `#SBATCH --gres=gpu:1` to `--gres=gpu:2`):

```bash
MUT_ID=$($PY -c "import yaml;print(yaml.safe_load(open('icml2027/configs/models.yaml'))['mutator']['hf_id'])")
if [ "$MUT_ID" = "$MODEL" ]; then
  export MUTATOR_BASE_URL="http://127.0.0.1:$PORT/v1"
else
  MPORT=$((PORT + 1))
  CUDA_VISIBLE_DEVICES=1 $PY -m vllm.entrypoints.openai.api_server --model "$MUT_ID" --port "$MPORT" \
    --trust-remote-code --max-model-len 8192 --gpu-memory-utilization 0.90 --enable-prefix-caching \
    > "$LOGDIR/vllm_mut_${SLURM_JOB_ID}.log" 2>&1 &
  MUT_PID=$!; trap 'kill $VLLM_PID $MUT_PID 2>/dev/null || true' EXIT
  for i in $(seq 1 180); do curl -sf "http://127.0.0.1:$MPORT/v1/models" >/dev/null 2>&1 && break; sleep 10; done
  export MUTATOR_BASE_URL="http://127.0.0.1:$MPORT/v1"
fi
```
and pin the evaluated model's server with `CUDA_VISIBLE_DEVICES=0` (or `0..TP-1`; then the mutator uses device `TP`).

- [ ] **Step 5: Run** `python -m pytest tests/optim/test_run_cell.py -q` → passed. **Step 6: Commit**

```bash
git add src/optim/run_cell.py tests/optim/test_run_cell.py icml2027/slurm/cell_group.sbatch
git commit -m "optim: idempotent cell runner with readout guard, budget check, frozen eval; mutator server in sbatch"
```

---

### Task 3: Queue (manifest expansion, submission, status)

**Files:**
- Create: `icml2027/scripts/queue.py`, `tests/optim/test_queue.py`

**Interfaces:**
- Produces: `expand(grid_yaml, models_yaml, panels_yaml, optimizers_dir) -> dict` (manifest `{"name","mutator","prereg_sha","cells":[...]}`; cell fields as in Task 2); CLI `queue.py submit --manifest <grid.yaml> [--max-concurrent 4] [--dry-run]` (writes `icml2027/queue/manifest.json`, creates `status.json = pending` for new cells, requeues `failed_oom` with attempt < 3 lowering `GPU_UTIL` by 0.05 per attempt, submits `sbatch --array=<clusters with unfinished cells>%<max_concurrent> --export=ALL,MODEL=…,SLUG=…,TP=…,DTYPE=…,GPU_UTIL=…,MANIFEST=icml2027/queue/manifest.json icml2027/slurm/cell_group.sbatch`, adding `--gres=gpu:<TP+1>`; records job ids in `icml2027/queue/state.json`); `queue.py status` (counts by state per model and overall, printed as a Markdown table and written to `state.json`).

- [ ] **Step 1: Write the failing test**

```python
# tests/optim/test_queue.py
import importlib.util, json
from pathlib import Path
spec = importlib.util.spec_from_file_location("queue_mod", "icml2027/scripts/queue.py"); Q = importlib.util.module_from_spec(spec); spec.loader.exec_module(Q)

def test_expand_tier1_counts():
    man = Q.expand("icml2027/configs/grid_tier1.yaml", "icml2027/configs/models.yaml", "icml2027/configs/panels.yaml", "icml2027/configs/optimizers")
    tier1 = [c for c in man["cells"] if c["fitness"] == "rps_cal"]
    assert len(tier1) == 10 * 6 * 4 * 3
    abl = [c for c in man["cells"] if c["fitness"] == "s0"]
    assert len(abl) == 2 * 4 * 3 and all(c["arm"] == "ga" for c in abl)
    ids = [c["cell_id"] for c in man["cells"]]; assert len(ids) == len(set(ids))
    assert all(c["budget_B"] == 80 for c in man["cells"])

def test_status_counts(tmp_path):
    for cid, s in [("a", "completed"), ("b", "failed_oom"), ("c", "completed")]:
        (tmp_path / cid).mkdir(); (tmp_path / cid / "status.json").write_text(json.dumps({"state": s}))
    assert Q.count_states(tmp_path) == {"completed": 2, "failed_oom": 1}
```

- [ ] **Step 2: Run** → FAIL

- [ ] **Step 3: Implement**

```python
# icml2027/scripts/queue.py
"""Expand a grid into cells, create their directories, submit Slurm arrays, report status.
Idempotent: re-running `submit` only schedules cells that are not completed."""
import argparse, collections, glob, json, os, subprocess, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
CELLS = ROOT / "icml2027/results/cells"; QDIR = ROOT / "icml2027/queue"

def _ld(p): return yaml.safe_load(open(ROOT / p if not Path(p).is_absolute() else p))

def expand(grid_yaml, models_yaml, panels_yaml, optimizers_dir):
    grid, models, panels = _ld(grid_yaml), _ld(models_yaml), _ld(panels_yaml); d = models["defaults"]
    by_slug = {m["slug"]: {**d, **m} for m in models["models"]}
    arm_cfg = {Path(f).stem: yaml.safe_load(open(f)) for f in glob.glob(str(ROOT / optimizers_dir / "*.yaml"))}
    common = {"panel_sizes": panels["panel_sizes"], "alphas": panels["alphas"], "exclude_globs": panels["exclude_case_ids_from"],
              "boot_B": panels["bootstrap"]["B"], "budget_B": grid["budget_B"]}
    def cell(arm, m, c, s, fitness):
        suffix = "" if fitness == "rps_cal" else f"-{fitness}"
        return {"cell_id": f"{arm}{suffix}__{m['slug']}__c{c}__s{s}", "arm": arm, "fitness": fitness, "model_slug": m["slug"],
                "hf_id": m["hf_id"], "prefill": m["prefill"], "tp": m["tp"], "dtype": m["dtype"],
                "gpu_memory_utilization": m["gpu_memory_utilization"], "cluster": c, "seed": s,
                "arm_cfg": arm_cfg[arm], **common}
    cells = [cell(a, m, c, s, grid["fitness"]) for m in by_slug.values() if m["tier"] in grid["models_tier"]
             for a in grid["arms"] for c in grid["clusters"] for s in grid["seeds"]]
    for ab in grid.get("ablations", []):
        cells += [cell(ab["arm"], by_slug[sl], c, s, ab["fitness"]) for sl in ab["models"] for c in ab["clusters"] for s in ab["seeds"]]
    try: prereg = subprocess.check_output(["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", "icml2027/PREREGISTRATION.md"], text=True).strip()
    except Exception: prereg = None
    return {"name": grid["name"], "mutator": models["mutator"], "prereg_sha": prereg, "slurm": grid["slurm"], "cells": cells}

def count_states(root=CELLS):
    c = collections.Counter()
    for f in Path(root).glob("*/status.json"): c[json.loads(f.read_text())["state"]] += 1
    return dict(c)

def _state(cid):
    f = CELLS / cid / "status.json"; return json.loads(f.read_text()) if f.exists() else None

def submit(grid_yaml, max_concurrent, dry):
    man = expand(grid_yaml, "icml2027/configs/models.yaml", "icml2027/configs/panels.yaml", "icml2027/configs/optimizers")
    QDIR.mkdir(parents=True, exist_ok=True); (QDIR / "manifest.json").write_text(json.dumps(man, indent=1))
    todo = collections.defaultdict(set); util = {}
    for c in man["cells"]:
        st = _state(c["cell_id"])
        if st is None:
            (CELLS / c["cell_id"]).mkdir(parents=True, exist_ok=True)
            (CELLS / c["cell_id"] / "status.json").write_text(json.dumps({"state": "pending", "attempt": 0}))
            st = {"state": "pending", "attempt": 0}
        if st["state"] in ("completed", "failed_readout", "running"): continue
        if st["state"] in ("failed_other", "stuck") or (st["state"] == "failed_oom" and st.get("attempt", 0) >= 3):
            (CELLS / c["cell_id"] / "status.json").write_text(json.dumps({**st, "state": "stuck"})); continue
        todo[c["model_slug"]].add(c["cluster"])
        util[c["model_slug"]] = min(util.get(c["model_slug"], 1.0), c["gpu_memory_utilization"] - 0.05 * (st["state"] == "failed_oom") * st.get("attempt", 0))
    state = json.loads((QDIR / "state.json").read_text()) if (QDIR / "state.json").exists() else {"jobs": []}
    for slug, clusters in todo.items():
        c0 = next(c for c in man["cells"] if c["model_slug"] == slug)
        arr = ",".join(map(str, sorted(clusters)))
        cmd = ["sbatch", "--parsable", f"--array={arr}%{max_concurrent}", f"--gres=gpu:{int(c0['tp']) + 1}",
               f"--export=ALL,MODEL={c0['hf_id']},SLUG={slug},TP={c0['tp']},DTYPE={c0['dtype']},GPU_UTIL={util[slug]:.2f},MANIFEST=icml2027/queue/manifest.json",
               "icml2027/slurm/cell_group.sbatch"]
        print(" ".join(cmd))
        if not dry:
            jid = subprocess.check_output(cmd, text=True, cwd=ROOT).strip(); state["jobs"].append({"slug": slug, "array": arr, "job_id": jid})
    state["counts"] = count_states(); (QDIR / "state.json").write_text(json.dumps(state, indent=1))

def status():
    per = collections.defaultdict(collections.Counter)
    for f in CELLS.glob("*/status.json"):
        per[f.parent.name.split("__")[1]][json.loads(f.read_text())["state"]] += 1
    states = sorted({s for c in per.values() for s in c})
    print("| model | " + " | ".join(states) + " |"); print("|---" * (len(states) + 1) + "|")
    for m, c in sorted(per.items()): print(f"| {m} | " + " | ".join(str(c.get(s, 0)) for s in states) + " |")
    total = count_states(); print("total:", total)
    st = json.loads((QDIR / "state.json").read_text()) if (QDIR / "state.json").exists() else {}
    st["counts"] = total; QDIR.mkdir(parents=True, exist_ok=True); (QDIR / "state.json").write_text(json.dumps(st, indent=1))

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit"); s.add_argument("--manifest", required=True); s.add_argument("--max-concurrent", type=int, default=4); s.add_argument("--dry-run", action="store_true")
    sub.add_parser("status"); a = ap.parse_args()
    submit(a.manifest, a.max_concurrent, a.dry_run) if a.cmd == "submit" else status()

if __name__ == "__main__":
    main()
```

Note on `--max-concurrent`: the `%N` limit is per array; with six model arrays in flight the global GPU count could exceed the 4-GPU policy in `AGENT.md`. Submit one model at a time (`--models <slug>` filter; add it as an argparse option that filters `todo`) or chain arrays with `--dependency=afterany:<previous>`; implement the dependency chain: pass `--dependency=afterany:{last_jid}` to every array after the first `max_concurrent // 2` arrays.

- [ ] **Step 4: Run** `python -m pytest tests/optim/test_queue.py -q` → 2 passed; `python icml2027/scripts/queue.py submit --manifest icml2027/configs/grid_tier1.yaml --dry-run` prints six sbatch lines. **Step 5: Commit**

```bash
git add icml2027/scripts/queue.py tests/optim/test_queue.py
git commit -m "icml2027: grid expansion, idempotent Slurm submission with OOM retry, status table"
```

---

### Task 4: Baselines and split-noise floor

**Files:**
- Create: `src/optim/baselines.py`, `tests/optim/test_baselines.py`

**Interfaces:**
- Produces: CLI `python -m src.optim.baselines --panels icml2027/configs/panels.yaml --out icml2027/results/baselines`; function `baseline_scores(panels) -> dict` with RPS and S₀ for: `cluster_mean` (point mass), `cluster_majority` (point mass at per-item mode), `global_mean` (point mass at the all-respondent rounded mean, computed on the train block of *all* clusters for that seed), `empirical_sample` (per-item empirical distribution as a forecast), `centroid_deterministic` (facet score of the cluster centroid → answer `1 + 4·score/100`, rounded, forward items; reversed on reverse-keyed items), `human` (from `frozen_eval.floors`), `decoder_m0` = `cluster_prior`. Writes `c{cluster}__s{seed}.json` and, over `baseline_redraws` redraws (seeds 101..120), `splitnoise_c{cluster}.json` with mean and SD per baseline.

- [ ] **Step 1: Write the failing test**

```python
# tests/optim/test_baselines.py
import numpy as np, pandas as pd
from src.optim.baselines import baseline_scores
from src.optim.panels import make_panels

def test_baselines_keys_and_prior_beats_constant_under_rps():
    rng = np.random.default_rng(0); n = 1500
    df = pd.DataFrame(rng.integers(1, 6, size=(n, 120)).astype(float), columns=[f"i{i}" for i in range(1, 121)])
    df["case"] = np.arange(n); df["clusters"] = 0
    b = baseline_scores(make_panels(df, 0, 1, {"train": 800, "opt": 40, "cal": 40, "eval": 200}))
    assert {"cluster_mean", "cluster_majority", "empirical_sample", "centroid_deterministic", "human", "cluster_prior"} <= set(b)
    assert b["cluster_prior"]["rps"] < b["cluster_mean"]["rps"]      # proper score: the distribution beats the constant
    assert b["cluster_mean"]["s0"] < b["empirical_sample"]["s0"]     # improper S0 (negated, lower better): the constant wins
```

- [ ] **Step 2: Run** → FAIL

- [ ] **Step 3: Implement**

```python
# src/optim/baselines.py
import argparse, yaml
from pathlib import Path
import numpy as np
from src.optim import scoring as S
from src.optim.frozen_eval import floors
from src.optim.orientation import REVERSE_MASK_120
from src.optim.panels import ITEM_SPLIT, FACET_OF_ITEM, answers, excluded_cases, load_corpus, make_panels
from src.optim.results import atomic_write_json

def _point(v, n, J):
    P = np.zeros((n, J, 5)); P[np.arange(n)[:, None], np.arange(J)[None, :], (np.clip(np.rint(v), 1, 5) - 1).astype(int)] = 1; return P

def _both(P, Y): return {"rps": float(np.nanmean(S.rps(P, Y))), "s0": float(np.nanmean(S.s0(P, Y)))}

def baseline_scores(panels):
    tgt = ITEM_SPLIT[1]; Ytr, Yev = answers(panels.train, tgt), answers(panels.eval, tgt); n, J = Yev.shape
    mean = np.broadcast_to(np.nanmean(Ytr, 0), (n, J))
    mode = np.broadcast_to(np.array([np.bincount(Ytr[:, j][~np.isnan(Ytr[:, j])].astype(int), minlength=6)[1:].argmax() + 1 for j in range(J)]), (n, J))
    emp = np.stack([np.array([np.mean(Ytr[:, j] == k) for k in range(1, 6)]) for j in range(J)])
    F0 = S.pop_cdf(Ytr)
    fac = {f: np.nanmean(answers(panels.train, [i for i in range(1, 121) if FACET_OF_ITEM[i] == f])) for f in set(FACET_OF_ITEM.values())}
    cent = np.array([fac[FACET_OF_ITEM[i]] for i in tgt])  # recoded scale; human answers are also recoded, so no flip here
    fl = floors(panels)
    return {"cluster_mean": _both(_point(mean, n, J), Yev), "cluster_majority": _both(_point(mode, n, J), Yev),
            "empirical_sample": _both(np.broadcast_to(emp, (n, J, 5)), Yev),
            "centroid_deterministic": _both(_point(np.broadcast_to(cent, (n, J)), n, J), Yev),
            "cluster_prior": _both(S.cdf_to_simplex(np.broadcast_to(F0, (n, J, 4))), Yev),
            "human": {"rps": fl["human_rps"], "s0": None}}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--panels", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.panels)); df = load_corpus(); ex = excluded_cases(cfg["exclude_case_ids_from"]); out = Path(a.out)
    for c in cfg["clusters"]:
        for s in cfg["seeds_headline"]:
            atomic_write_json(out / f"c{c}__s{s}.json", baseline_scores(make_panels(df, c, s, cfg["panel_sizes"], ex)))
        red = [baseline_scores(make_panels(df, c, 100 + r, cfg["panel_sizes"], ex)) for r in range(1, cfg["baseline_redraws"] + 1)]
        atomic_write_json(out / f"splitnoise_c{c}.json", {k: {"rps_mean": float(np.mean([r[k]["rps"] for r in red])),
                                                              "rps_sd": float(np.std([r[k]["rps"] for r in red], ddof=1))} for k in red[0]})

if __name__ == "__main__":
    main()
```

`centroid_deterministic` note: the facet mean of the cluster's train block is on the corpus's recoded scale, and so are the targets, so it is compared directly; the reviewers' "centroid-derived answers" baseline is exactly this.

- [ ] **Step 4: Run** → passed. **Step 5: Commit**

```bash
git add src/optim/baselines.py tests/optim/test_baselines.py
git commit -m "optim: non-LLM baselines on the frozen panels and the 20-redraw split-noise floor"
```

---

### Task 5: Aggregation, bootstrap + BH, m* pricing, figures, manifest, regenerate script

**Files:**
- Create: `icml2027/scripts/analysis/{aggregate.py,bootstrap.py,price_mstar.py,figures.py,manifest.py,regenerate_all.sh}`, `tests/optim/test_analysis.py`

**Interfaces:**
- `aggregate.py --results R --out O`: reads every `cells/*/eval_frozen.json` + `run_meta.json`; writes `tier1_cells.csv` (one row per cell: ids, base/evolved rps_cal, delta, CI, alpha, vr_between, wrong_persona, floors, mass, excluded) and `evaluations_to_threshold.csv` (from `history.jsonl`: first `eval_idx` with `rps_cal_opt ≤ base − 0.005`, else NaN).
- `bootstrap.py --results R --panels P --out O`: per cell, paired bootstrap of evolved vs each of {base, random, paraphrase, bestofb} using `per_respondent.json` of the matched cells (same model, cluster, seed); BH across all tests; writes `tests_bh.csv` and `tier1_summary.csv` (per arm: mean Δ over models with bootstrap-over-cells CI, count of cells significant vs base / vs random / vs floor `cluster_prior_rps`).
- `price_mstar.py --results R --panels P --out O`: for every completed cell, loads `belief_probs.npz` (base and evolved eval beliefs; `run_cell` must save them: add `np.savez_compressed(w.dir / "belief_probs.npz", base=Pb, evolved=Pe)` to `run_one`, returning `Pb` from `evaluate_frozen` too) and calls `h26_answer_equivalent`'s decoder logic (import `fit_decoder`, `BUDGETS` from `arr2026/scripts/hyp/h26_answer_equivalent.py`); writes `mstar_by_arm_model.csv` and fills `eval_frozen.json["mstar"]` via an `mstar.json` sidecar (do not rewrite `eval_frozen.json`).
- `figures.py --aggregates A --out F`: fig1 Δm\* (or ΔRPS_cal) per arm × model with CIs and floor/ceiling lines; fig2 Pareto fronts from `pareto_front.json`; fig3 evaluations-to-threshold; fig4 cross-model transfer heatmap (from `crossmodel_matrix.csv` if present). Matplotlib, PDF, colour-blind-safe palette, one function per figure.
- `manifest.py --results R`: writes `MANIFEST.sha256` over tracked files.
- `regenerate_all.sh`: runs the five scripts in order, `set -euo pipefail`.

- [ ] **Step 1: Write the failing test**

```python
# tests/optim/test_analysis.py
import json, subprocess, sys
import numpy as np

def _fake_cell(root, cid, delta, base=0.16):
    d = root / "cells" / cid; d.mkdir(parents=True)
    ev = {"panel": {"cluster": 0, "seed": 1}, "base": {"rps_cal": base, "alpha": .5, "vr_between": .2, "mass_on_scale": .99},
          "evolved": {"rps_cal": base + delta, "alpha": .7, "vr_between": .25, "mass_on_scale": .99, "wrong_persona_rps_cal": base + .01},
          "delta": {"rps_cal": delta, "rps_cal_ci95": [delta - .002, delta + .002]}, "floors": {"cluster_prior_rps": .155, "cluster_mean_rps": .17, "human_rps": .2},
          "excluded": False}
    (d / "eval_frozen.json").write_text(json.dumps(ev)); (d / "status.json").write_text(json.dumps({"state": "completed"}))
    arm, slug, c, s = cid.split("__")
    (d / "run_meta.json").write_text(json.dumps({"arm": arm, "model_slug": slug, "cluster": int(c[1:]), "seed": int(s[1:])}))
    (d / "history.jsonl").write_text("\n".join(json.dumps({"eval_idx": i, "rps_cal_opt": base - 0.001 * i}) for i in range(10)))
    rng = np.random.default_rng(0); b = rng.normal(base, .01, 200)
    (d / "per_respondent.json").write_text(json.dumps({"base_rps_cal": b.tolist(), "evolved_rps_cal": (b + delta).tolist()}))

def test_aggregate_and_bootstrap(tmp_path):
    for arm, dl in [("ga", -.005), ("random", -.001), ("paraphrase", 0.0), ("bestofb", -.002)]:
        _fake_cell(tmp_path, f"{arm}__m__c0__s1", dl)
    out = tmp_path / "agg"
    subprocess.check_call([sys.executable, "icml2027/scripts/analysis/aggregate.py", "--results", str(tmp_path), "--out", str(out)])
    subprocess.check_call([sys.executable, "icml2027/scripts/analysis/bootstrap.py", "--results", str(tmp_path), "--panels", "icml2027/configs/panels.yaml", "--out", str(out)])
    import pandas as pd
    cells = pd.read_csv(out / "tier1_cells.csv"); assert len(cells) == 4
    tests = pd.read_csv(out / "tests_bh.csv"); row = tests[(tests.arm == "ga") & (tests.comparator == "random")].iloc[0]
    assert row.delta < 0 and row.q_bh < 0.05
    e2t = pd.read_csv(out / "evaluations_to_threshold.csv"); assert int(e2t[e2t.arm == "ga"].evals.iloc[0]) == 5
```

- [ ] **Step 2: Run** → FAIL

- [ ] **Step 3: Implement `aggregate.py`**

```python
#!/usr/bin/env python
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    R, O = Path(a.results), Path(a.out); O.mkdir(parents=True, exist_ok=True); rows, e2t = [], []
    for d in sorted((R / "cells").glob("*")):
        f = d / "eval_frozen.json"
        if not f.exists(): continue
        ev, meta = json.loads(f.read_text()), json.loads((d / "run_meta.json").read_text())
        rows.append({"cell_id": d.name, "arm": meta["arm"], "model": meta["model_slug"], "cluster": meta["cluster"], "seed": meta["seed"],
                     "fitness": meta.get("fitness", "rps_cal"), "base_rps_cal": ev["base"]["rps_cal"], "evolved_rps_cal": ev["evolved"]["rps_cal"],
                     "delta": ev["delta"]["rps_cal"], "ci_lo": ev["delta"]["rps_cal_ci95"][0], "ci_hi": ev["delta"]["rps_cal_ci95"][1],
                     "alpha_base": ev["base"]["alpha"], "alpha_evolved": ev["evolved"]["alpha"], "vr_between": ev["evolved"]["vr_between"],
                     "wrong_persona": ev["evolved"].get("wrong_persona_rps_cal"), "mass": ev["evolved"]["mass_on_scale"], "excluded": ev["excluded"],
                     **{f"floor_{k}": v for k, v in ev["floors"].items()}})
        base = ev["base"]["rps_cal"]; hist = [json.loads(l) for l in (d / "history.jsonl").read_text().splitlines() if l.strip()]
        hit = next((h["eval_idx"] for h in hist if h["rps_cal_opt"] <= base - 0.005), np.nan)
        e2t.append({"cell_id": d.name, "arm": meta["arm"], "model": meta["model_slug"], "evals": hit})
    pd.DataFrame(rows).to_csv(O / "tier1_cells.csv", index=False); pd.DataFrame(e2t).to_csv(O / "evaluations_to_threshold.csv", index=False)

if __name__ == "__main__":
    main()
```

Note: `evaluations_to_threshold` compares search-panel fitness to the frozen base score only as a common reference line; label it so in the figure caption.

- [ ] **Step 4: Implement `bootstrap.py`**

```python
#!/usr/bin/env python
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, yaml

def boot(d, B, seed):
    rng = np.random.default_rng(seed); n = len(d); bs = np.array([np.nanmean(d[rng.integers(0, n, n)]) for _ in range(B)])
    p = 2 * min(np.mean(bs >= 0), np.mean(bs <= 0)); return float(np.nanmean(d)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), max(p, 1.0 / B)

def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = np.empty_like(p, float); m = len(p); prev = 1.0
    for rank, i in reversed(list(enumerate(o, 1))): prev = min(prev, p[i] * m / rank); q[i] = prev
    return q

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True); ap.add_argument("--panels", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.panels)); B, seed = cfg["bootstrap"]["B"], cfg["bootstrap"]["seed"]; R, O = Path(a.results), Path(a.out)
    per = {}
    for d in (R / "cells").glob("*"):
        if (d / "per_respondent.json").exists():
            arm, slug, c, s = d.name.split("__"); per[(arm, slug, c, s)] = json.loads((d / "per_respondent.json").read_text())
    rows = []
    for (arm, slug, c, s), v in per.items():
        ev = np.array(v["evolved_rps_cal"])
        comps = {"base": np.array(v["base_rps_cal"])}
        for comp in ("random", "paraphrase", "bestofb"):
            if comp != arm and (comp, slug, c, s) in per: comps[comp] = np.array(per[(comp, slug, c, s)]["evolved_rps_cal"])
        for comp, x in comps.items():
            m, lo, hi, p = boot(ev - x, B, seed); rows.append({"arm": arm, "model": slug, "cluster": c, "seed": s, "comparator": comp, "delta": m, "lo": lo, "hi": hi, "p": p})
    t = pd.DataFrame(rows); t["q_bh"] = bh(t["p"].values) if len(t) else []
    t.to_csv(O / "tests_bh.csv", index=False)
    summ = []
    for arm, g in t.groupby("arm"):
        vb = g[g.comparator == "base"]; d = vb["delta"].to_numpy()
        m, lo, hi, _ = boot(d, B, seed) if len(d) > 1 else (float(d.mean()), np.nan, np.nan, np.nan)
        summ.append({"arm": arm, "cells": len(vb), "mean_delta_vs_base": m, "lo": lo, "hi": hi,
                     **{f"sig_better_vs_{k}": int(((g.comparator == k) & (g.q_bh < cfg["bh_q"]) & (g.delta < 0)).sum()) for k in ("base", "random", "paraphrase", "bestofb")}})
    pd.DataFrame(summ).to_csv(O / "tier1_summary.csv", index=False)

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Implement `price_mstar.py`, `figures.py`, `manifest.py`, `regenerate_all.sh`**

```python
#!/usr/bin/env python
# icml2027/scripts/analysis/price_mstar.py
"""Answer-equivalent of base vs evolved personas per cell, reusing the audit's decoder protocol."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "arr2026/scripts/hyp"))
from src.optim import scoring as S
from src.optim.panels import ITEM_SPLIT, answers, excluded_cases, load_corpus, make_panels
from src.optim.results import atomic_write_json

BUDGETS = [0, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60]
LAMS = [0.1, 1, 10, 100, 1000]

def onehot(X):
    X = np.nan_to_num(X, nan=3).astype(int); return np.concatenate([(X == k).astype(float) for k in range(1, 6)], axis=1)

def decoder_curve(panels, rng, reps=5):
    """RPS of a ridge decoder given m observed input answers, penalty retuned per m on the calibration panel."""
    inp, tgt = ITEM_SPLIT; Xtr, Ytr = answers(panels.train, inp), answers(panels.train, tgt)
    T = np.stack([(Ytr <= t).astype(float) for t in range(1, 5)], axis=-1).reshape(len(Ytr), -1)
    out = {}
    for m in BUDGETS:
        vals = []
        for _ in range(reps if m else 1):
            cols = rng.choice(len(inp), size=m, replace=False) if m else np.array([], int)
            def feat(df): return np.concatenate([np.ones((len(df), 1)), onehot(answers(df, [inp[c] for c in cols]))], axis=1) if m else np.ones((len(df), 1))
            Z = feat(panels.train)
            def pred(df, lam):
                W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ T); F = S.iso_project((feat(df) @ W).reshape(len(df), len(tgt), 4))
                return S.cdf_to_simplex(F)
            lam = min(LAMS, key=lambda l: np.nanmean(S.rps(pred(panels.cal, l), answers(panels.cal, tgt))))
            vals.append(float(np.nanmean(S.rps(pred(panels.eval, lam), answers(panels.eval, tgt)))))
        out[m] = float(np.mean(vals))
    return out

def interp_mstar(curve, score):
    ms = sorted(curve)
    if curve[ms[0]] <= score: return 0.0
    for a, b in zip(ms, ms[1:]):
        if curve[b] <= score: return a + (b - a) * (curve[a] - score) / (curve[a] - curve[b])
    return float(ms[-1])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True); ap.add_argument("--panels", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.panels)); df = load_corpus(); ex = excluded_cases(cfg["exclude_case_ids_from"]); curves = {}; rows = []
    for d in sorted(Path(a.results, "cells").glob("*")):
        f = d / "eval_frozen.json"
        if not f.exists(): continue
        ev = json.loads(f.read_text()); c, s = ev["panel"]["cluster"], ev["panel"]["seed"]
        if (c, s) not in curves: curves[(c, s)] = decoder_curve(make_panels(df, c, s, cfg["panel_sizes"], ex), np.random.default_rng(s))
        cur = curves[(c, s)]; mb, me = interp_mstar(cur, ev["base"]["rps_cal"]), interp_mstar(cur, ev["evolved"]["rps_cal"])
        atomic_write_json(d / "mstar.json", {"curve": cur, "base_interp": mb, "evolved_interp": me, "delta_interp": me - mb})
        arm, slug, _, _ = d.name.split("__"); rows.append({"cell_id": d.name, "arm": arm, "model": slug, "cluster": c, "seed": s, "mstar_base": mb, "mstar_evolved": me, "delta": me - mb})
    pd.DataFrame(rows).to_csv(Path(a.out) / "mstar_by_arm_model.csv", index=False)

if __name__ == "__main__":
    main()
```

Before trusting `price_mstar.py`, reproduce one audit number: run `decoder_curve` on the audit's IPIP panel for Qwen3.6-35B-A3B and check `interp_mstar` lands within ±0.3 of the published 3.07 (use the calibrated score from `arr2026/results_euler/h26_equiv/`). If it does not, import the audit's `fit_decoder` from `h26_answer_equivalent.py` instead of this re-implementation.

```python
#!/usr/bin/env python
# icml2027/scripts/analysis/figures.py
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd

OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000", "#999999", "#882255"]

def fig1(A, F):
    m = pd.read_csv(A / "mstar_by_arm_model.csv") if (A / "mstar_by_arm_model.csv").exists() else None
    c = pd.read_csv(A / "tier1_cells.csv"); y, lab = ("delta", "Δ m* (answers)") if m is not None else ("delta", "Δ RPS_cal (lower is better)")
    d = (m if m is not None else c).groupby(["arm", "model"])[y].agg(["mean", "sem"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 3.2)); arms = sorted(d.arm.unique()); models = sorted(d.model.unique())
    for k, mod in enumerate(models):
        s = d[d.model == mod].set_index("arm").reindex(arms)
        ax.errorbar([i + 0.08 * (k - len(models) / 2) for i in range(len(arms))], s["mean"], yerr=1.96 * s["sem"], fmt="o", color=OKABE[k % 10], label=mod, ms=3)
    ax.axhline(0, color="grey", lw=0.8); ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms, rotation=30); ax.set_ylabel(lab)
    ax.legend(fontsize=6, ncol=3); fig.tight_layout(); fig.savefig(F / "fig1.pdf")

def fig2(R, F):
    fig, ax = plt.subplots(figsize=(4, 3))
    for k, f in enumerate(sorted(Path(R, "cells").glob("*/pareto_front.json"))[:12]):
        pts = sorted((p["objectives"] for p in json.loads(f.read_text())), key=lambda o: o[0])
        ax.plot([p[0] for p in pts], [-p[1] for p in pts], "-o", ms=2, color=OKABE[k % 10], lw=0.8)
    ax.set_xlabel("RPS_cal (lower better)"); ax.set_ylabel("VR_between (higher = more individuation)"); fig.tight_layout(); fig.savefig(F / "fig2.pdf")

def fig3(A, F):
    e = pd.read_csv(A / "evaluations_to_threshold.csv"); fig, ax = plt.subplots(figsize=(5, 3))
    arms = sorted(e.arm.unique()); ax.boxplot([e[e.arm == a].evals.dropna() for a in arms], labels=arms)
    ax.set_ylabel("evaluations to base − 0.005"); plt.xticks(rotation=30); fig.tight_layout(); fig.savefig(F / "fig3.pdf")

def fig4(A, F):
    p = A / "crossmodel_matrix.csv"
    if not p.exists(): return
    M = pd.read_csv(p, index_col=0); fig, ax = plt.subplots(figsize=(4, 3.5)); im = ax.imshow(M.values, cmap="viridis")
    ax.set_xticks(range(len(M.columns))); ax.set_xticklabels(M.columns, rotation=60, fontsize=6); ax.set_yticks(range(len(M.index))); ax.set_yticklabels(M.index, fontsize=6)
    fig.colorbar(im, ax=ax, label="share of ΔRPS_cal retained"); fig.tight_layout(); fig.savefig(F / "fig4.pdf")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--aggregates", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    A, F = Path(a.aggregates), Path(a.out); F.mkdir(parents=True, exist_ok=True)
    fig1(A, F); fig2(A.parent, F); fig3(A, F); fig4(A, F)

if __name__ == "__main__":
    main()
```

```python
#!/usr/bin/env python
# icml2027/scripts/analysis/manifest.py
import argparse, hashlib
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True); a = ap.parse_args(); R = Path(a.results)
lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(R)}" for p in sorted(R.rglob("*"))
         if p.is_file() and p.suffix in {".json", ".jsonl", ".csv", ".yaml", ".md", ".pdf"} and p.name != "MANIFEST.sha256"]
(R / "MANIFEST.sha256").write_text("\n".join(lines) + "\n"); print(len(lines), "files")
```

```bash
#!/bin/bash
# icml2027/scripts/analysis/regenerate_all.sh — every aggregate from results/, deterministic.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-python}
$PY -m src.optim.baselines --panels icml2027/configs/panels.yaml --out icml2027/results/baselines
$PY icml2027/scripts/analysis/aggregate.py --results icml2027/results --out icml2027/results/aggregates
$PY icml2027/scripts/analysis/bootstrap.py --results icml2027/results --panels icml2027/configs/panels.yaml --out icml2027/results/aggregates
$PY icml2027/scripts/analysis/price_mstar.py --results icml2027/results --panels icml2027/configs/panels.yaml --out icml2027/results/aggregates
$PY icml2027/scripts/analysis/figures.py --aggregates icml2027/results/aggregates --out icml2027/results/aggregates/figures
$PY icml2027/scripts/analysis/manifest.py --results icml2027/results
echo REGENERATED
```

- [ ] **Step 6: Run** `python -m pytest tests/optim/test_analysis.py -q` → passed. **Step 7: Commit**

```bash
chmod +x icml2027/scripts/analysis/regenerate_all.sh
git add icml2027/scripts/analysis tests/optim/test_analysis.py
git commit -m "icml2027: aggregation, paired bootstrap with BH, m* pricing, figures, manifest, regenerate script"
```

---

### Task 6: Transfer and cross-model evaluations (E6, H6)

**Files:**
- Create: `src/optim/transfer.py`, `icml2027/slurm/transfer.sbatch`, `tests/optim/test_transfer.py`

**Interfaces:**
- `python -m src.optim.transfer ipip300 --cells-glob 'icml2027/results/cells/*__<slug>__*'`: for each completed cell of the served model, build personas from the IPIP-NEO-120 short form of the 300 `Test-set.json` respondents (reuse `arr2026/scripts/hyp/h24_ipip300_readout.py`'s `load_corpus` for items, keys and orientation), read beliefs on the 180 unseen items for base and evolved genotypes, score RPS_cal with α selected on a 100-respondent calibration split, evaluate on the other 200; write `transfer_ipip300.json` in the cell dir.
- `python -m src.optim.transfer crossmodel --source-slug A --target-slug B`: evaluate cell A's `best_genotype.json` with model B's readout on A's frozen panels via `evaluate_frozen`; write `icml2027/results/crossmodel/<A>__to__<B>__c{c}__s{s}.json`; `aggregate.py` gains a `crossmodel_matrix.csv` builder (share of ΔRPS_cal retained = Δ_transfer / Δ_source).
- `transfer.sbatch` serves one model and runs both subcommands for it.

- [ ] **Step 1: Write the failing test** (retention computation only; the GPU paths are exercised in E0)

```python
# tests/optim/test_transfer.py
from src.optim.transfer import retention
def test_retention():
    assert retention(-0.004, -0.002) == 0.5 and retention(0.0, -0.001) is None
```

- [ ] **Step 2–4:** implement `retention(delta_source, delta_transfer) -> float | None` (None when |delta_source| < 1e-6), the two subcommands reusing `frozen_eval._score_arm` and `h24_ipip300_readout.load_corpus`, and the sbatch (copy the server block from `cell_group.sbatch`, then `$PY -m src.optim.transfer ipip300 …` and a loop over target slugs). Run the unit test → passed.

- [ ] **Step 5: Commit**

```bash
git add src/optim/transfer.py icml2027/slurm/transfer.sbatch tests/optim/test_transfer.py
git commit -m "optim: IPIP-NEO-300 unseen-item transfer and cross-model transfer (H4, H6)"
```

---

### Task 7: E0 pilot on Euler (first real run)

**Files:**
- Create: `icml2027/configs/grid_pilot.yaml`
- Modify: `icml2027/STATUS.md`

- [ ] **Step 1: Pilot grid**

```yaml
# icml2027/configs/grid_pilot.yaml
name: pilot
budget_B: 80
fitness: rps_cal
arms: [ga, gepa, random]
models_tier: [1]
clusters: [0]
seeds: [1, 2, 3]
slurm: {partition: infer, time: "12:00:00", cpus: 16, mem: 200G, max_concurrent_arrays: 1}
```
Add a `--models` filter to `queue.py submit` (Task 3 note) and submit only `Qwen_Qwen3p6-35B-A3B`.

- [ ] **Step 2: Sync and run on Euler**

```bash
git push origin icml2027
ssh airi-h200 'cd /home/glazkov/personality-twins-arr/LLM-PersonaBench && git fetch origin && git checkout icml2027 && git pull --ff-only && sbatch icml2027/slurm/preflight.sbatch 1'
# after preflight succeeds:
ssh airi-h200 'cd /home/glazkov/personality-twins-arr/LLM-PersonaBench && /home/glazkov/personality-twins-arr/vllmenv/bin/python icml2027/scripts/queue.py submit --manifest icml2027/configs/grid_pilot.yaml --models Qwen_Qwen3p6-35B-A3B --max-concurrent 1'
```

- [ ] **Step 3: Record throughput and Gate G1**

From `run_meta.json` of the nine pilot cells: wall-clock per cell, forward passes, tokens. From `eval_frozen.json`: ΔRPS_cal with CI for GA and GEPA vs base, and vs random. Run `price_mstar.py` on the pilot cells. Write both into `icml2027/STATUS.md` (gate table + journal) with commit hashes. G1 passes only if GA or GEPA shows ΔRPS_cal < 0 with the CI excluding 0 in ≥ 2 of 3 seeds and a positive Δm\*. If it fails, stop and ask the user (AGENT.md §7).

- [ ] **Step 4: Commit results and status**

```bash
git add icml2027/results/cells icml2027/queue icml2027/STATUS.md icml2027/configs/grid_pilot.yaml
git commit -m "icml2027: E0 pilot results and Gate G1 decision"
git push origin icml2027
```

---

## Self-review

- Spec coverage: E3 baselines (Task 4) incl. constant floor, majority, global mean, empirical sampling, centroid-deterministic, human ceiling, prior ✓; LLM baselines via arms (plan 2) ✓; wrong-persona control (Task 1) ✓; E4 seeds/splits/bootstrap/BH (Tasks 3, 5) ✓; E6 transfer and H6 cross-model (Task 6) ✓; E7 `m*` (Task 5) ✓; E8 mutator sensitivity = rerun grid with `mutator.hf_id` changed, cell ids suffixed `-mut<slug>` (add to `queue.expand` when E8 is scheduled); E0 pilot (Task 7) ✓. The global-mean baseline needs the all-cluster train block; `baseline_scores` currently computes cluster-local baselines only — add `global_mean` by passing a second `Panels` built with `clusters` ignored (`df.assign(clusters=cluster)`), in Task 4 Step 3, before G2.
- Types: `evaluate_frozen` returns the `eval_frozen.json` schema of `results/README.md` plus `per_respondent`, popped by `run_one`; `run_one` saves `belief_probs.npz` (Task 5 note) — make that change in Task 2's implementation directly.
- The analysis scripts read only `results/`; `price_mstar.py` also reads the corpus to rebuild panels deterministically from `(cluster, seed)`, which `run_meta.json` records.
