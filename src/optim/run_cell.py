"""Run every cell of one (model, cluster) from the manifest against one served model.
Idempotent: completed cells are skipped; a failed attempt is archived under attempts/<n>/."""
import argparse
import json
import os
import platform
import subprocess
import sys
import traceback
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


def git_sha():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        p = ROOT / "icml2027/queue/DEPLOYED_SHA"
        return p.read_text().strip() if p.exists() else "unknown"


def run_one(cell, df, readout, mutator, results_root, meta_extra, panels_cache=None) -> str:
    w = CellWriter(results_root, cell["cell_id"])
    st = w.dir / "status.json"
    prev = json.loads(st.read_text()) if st.exists() else {}
    if prev.get("state") in ("completed", "failed_readout"):
        return prev["state"]
    attempt = int(prev.get("attempt", 0)) + 1
    if (w.dir / "history.jsonl").exists() or (w.dir / "eval_frozen.json").exists():
        w.archive_attempt()
    w.set_status("running", attempt=attempt, slurm_job_id=meta_extra.get("slurm_job_id"))
    w.write_config(cell)
    meta = {**meta_extra, **{k: cell[k] for k in ("cell_id", "arm", "fitness", "model_slug", "hf_id", "cluster", "seed",
                                                 "budget_B")}, "started": now()}
    try:
        key = (cell["cluster"], cell["seed"])
        if panels_cache is not None and key in panels_cache:
            panels = panels_cache[key]
        else:
            panels = make_panels(df, cell["cluster"], cell["seed"], cell["panel_sizes"],
                                 excluded_cases(cell.get("exclude_globs", [])))
            if panels_cache is not None:
                panels_cache[key] = panels
        rng = np.random.default_rng(cell["seed"] * 7919 + cell["cluster"])
        fit = Fitness(panels, readout, cell["alphas"], rng, objective=cell.get("fitness", "rps_cal"))
        seed_g = seed_genotype(cell["cluster"])
        # readout guard on 8 optimisation respondents; not charged, never used for selection
        guard = fit.evaluate_subset(seed_g, np.arange(min(8, fit.n_opt)))
        meta["mass_on_scale_guard"] = guard.mass
        meta["readout_guard_passed"] = bool(guard.mass >= 0.5)
        if not meta["readout_guard_passed"]:
            w.write_meta(meta)
            w.set_status("failed_readout", attempt=attempt, message=f"mass {guard.mass:.3f}")
            return "failed_readout"
        arm = ARMS[cell["arm"]](cell["arm_cfg"], rng, mutator, seed_g)
        ledger = BudgetLedger(cell["budget_B"])
        best = arm.run(fit, seed_g, ledger, w)
        meta["mass_on_scale_mean"] = arm.history[0]["mass_on_scale"]
        if abs(ledger.used_f - cell["budget_B"]) > 1e-6:
            raise RuntimeError(f"budget not exhausted: {ledger.used_f}/{cell['budget_B']}")
        ev = evaluate_frozen(panels, readout, seed_g, best, cell["alphas"], boot_B=cell.get("boot_B", 2000))
        bel = ev.pop("_beliefs")
        np.savez_compressed(w.dir / "belief_probs.npz", **bel)
        atomic_write_json(w.dir / "per_respondent.json", ev.pop("per_respondent"))
        w.write_eval(ev)
        meta.update(ledger.to_dict())
        meta["finished"] = now()
        w.write_meta(meta)
        w.set_status("completed", attempt=attempt)
        return "completed"
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()[-6000:]
        low = tb.lower()
        state = "failed_oom" if ("out of memory" in low or "connection refused" in low or "apiconnectionerror" in low) \
            else "failed_other"
        meta["error"] = tb
        w.write_meta(meta)
        w.set_status(state, attempt=attempt, message=f"{type(e).__name__}: {e}"[:500])
        print(tb, file=sys.stderr, flush=True)
        return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model-slug", required=True)
    ap.add_argument("--cluster", type=int, required=True)
    ap.add_argument("--slurm-job-id", default="local")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these cell ids")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="restrict to these seeds")
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    cells = [c for c in man["cells"] if c["model_slug"] == a.model_slug and c["cluster"] == a.cluster
             and (not a.only or c["cell_id"] in a.only) and (not a.seeds or c["seed"] in a.seeds)]
    if not cells:
        print("no cells")
        sys.exit(0)
    from src.optim.readout import VLLMReadout
    readout = VLLMReadout(cells[0]["hf_id"], prefill=cells[0].get("prefill", "My answer is "), workers=a.workers)
    mutator = LLMMutator(man["mutator"]["hf_id"], base_url=os.environ.get("MUTATOR_BASE_URL"))
    df = load_corpus()
    meta = {"slurm_job_id": a.slurm_job_id, "node": platform.node(), "git_sha": git_sha(),
            "prereg_sha": man.get("prereg_sha"), "mutator": man["mutator"]["hf_id"], "readout": "belief_vllm_completions",
            "gpu": os.environ.get("GPU_NAME", ""), "tp": cells[0].get("tp"), "dtype": cells[0].get("dtype")}
    states, cache = [], {}
    for c in sorted(cells, key=lambda c: (c["seed"], c["arm"])):
        s = run_one(c, df, readout, mutator, RESULTS, meta, panels_cache=cache)
        states.append(s)
        print(f"{now()} {c['cell_id']}: {s}", flush=True)
    sys.exit(0 if all(s in ("completed", "failed_readout") for s in states) else 1)


if __name__ == "__main__":
    main()
