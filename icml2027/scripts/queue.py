#!/usr/bin/env python
"""Expand a grid into cells, create their directories, submit ONE Slurm array over (model, cluster)
groups with a global concurrency limit, and report status. Idempotent."""
import argparse
import collections
import glob
import json
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CELLS = ROOT / "icml2027/results/cells"
QDIR = ROOT / "icml2027/queue"


def _ld(p):
    p = Path(p)
    return yaml.safe_load(open(p if p.is_absolute() else ROOT / p))


def expand(grid_yaml, models_yaml="icml2027/configs/models.yaml", panels_yaml="icml2027/configs/panels.yaml",
           optimizers_dir="icml2027/configs/optimizers"):
    grid, models, panels = _ld(grid_yaml), _ld(models_yaml), _ld(panels_yaml)
    d = models["defaults"]
    by_slug = {m["slug"]: {**d, **m} for m in models["models"]}
    arm_cfg = {Path(f).stem: yaml.safe_load(open(f)) for f in glob.glob(str(ROOT / optimizers_dir / "*.yaml"))}
    common = {"panel_sizes": panels["panel_sizes"], "alphas": panels["alphas"],
              "exclude_globs": panels["exclude_case_ids_from"], "boot_B": panels["bootstrap"]["B"],
              "budget_B": grid["budget_B"]}
    only = set(grid.get("models", []) or [])

    def cell(arm, m, c, s, fitness):
        suffix = "" if fitness == "rps_cal" else f"-{fitness}"
        return {"cell_id": f"{arm}{suffix}__{m['slug']}__c{c}__s{s}", "arm": arm, "fitness": fitness,
                "model_slug": m["slug"], "hf_id": m["hf_id"], "prefill": m["prefill"], "tp": m["tp"],
                "dtype": m["dtype"], "gpu_memory_utilization": m["gpu_memory_utilization"], "cluster": c, "seed": s,
                "arm_cfg": arm_cfg[arm], **common}

    cells = [cell(a, m, c, s, grid["fitness"]) for m in by_slug.values()
             if m["tier"] in grid["models_tier"] and (not only or m["slug"] in only)
             for a in grid["arms"] for c in grid["clusters"] for s in grid["seeds"]]
    for ab in grid.get("ablations", []) or []:
        cells += [cell(ab["arm"], by_slug[sl], c, s, ab["fitness"]) for sl in ab["models"] for c in ab["clusters"]
                  for s in ab["seeds"]]
    try:
        prereg = subprocess.check_output(["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--",
                                          "icml2027/PREREGISTRATION.md"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        prereg = (QDIR / "PREREG_SHA").read_text().strip() if (QDIR / "PREREG_SHA").exists() else None
    return {"name": grid["name"], "mutator": models["mutator"], "prereg_sha": prereg, "cells": cells}


def count_states(root=CELLS):
    c = collections.Counter()
    for f in Path(root).glob("*/status.json"):
        c[json.loads(f.read_text())["state"]] += 1
    return dict(c)


def _state(cid):
    f = CELLS / cid / "status.json"
    return json.loads(f.read_text()) if f.exists() else None


def submit(grid_yaml, max_concurrent, dry, after=None, reset_running=False):
    man = expand(grid_yaml)
    QDIR.mkdir(parents=True, exist_ok=True)
    mpath = QDIR / f"manifest_{man['name']}.json"
    mpath.write_text(json.dumps(man, indent=1))
    groups = collections.OrderedDict()
    busy = {(c["model_slug"], c["cluster"]) for c in man["cells"]
            if not reset_running and (_state(c["cell_id"]) or {}).get("state") == "running"}
    for c in man["cells"]:
        if (c["model_slug"], c["cluster"]) in busy:
            continue
        st = _state(c["cell_id"])
        if st is None:
            (CELLS / c["cell_id"]).mkdir(parents=True, exist_ok=True)
            (CELLS / c["cell_id"] / "status.json").write_text(json.dumps({"state": "pending", "attempt": 0}))
            st = {"state": "pending", "attempt": 0}
        s = st["state"]
        if s == "running" and reset_running:
            s = "pending"
        if s in ("completed", "failed_readout", "running", "stuck"):
            continue
        if s == "failed_other" and st.get("attempt", 0) >= 2 or s == "failed_oom" and st.get("attempt", 0) >= 3:
            (CELLS / c["cell_id"] / "status.json").write_text(json.dumps({**st, "state": "stuck"}))
            continue
        groups.setdefault((c["model_slug"], c["cluster"]), c)
    single = [(k, c) for k, c in groups.items() if int(c["tp"]) == 1]
    multi = [(k, c) for k, c in groups.items() if int(c["tp"]) > 1]
    jobs = []
    for label, gl in (("tp1", single), ("tpN", multi)):
        if not gl:
            continue
        gfile = QDIR / f"groups_{man['name']}_{label}.json"
        gfile.write_text(json.dumps([{"hf_id": c["hf_id"], "slug": k[0], "cluster": k[1], "tp": c["tp"],
                                      "dtype": c["dtype"], "gpu_util": c["gpu_memory_utilization"],
                                      "split": 1 if "Qwen3p6" in k[0] else 2} for k, c in gl],
                                    indent=1))
        gpus = max(int(c["tp"]) for _, c in gl)
        conc = max_concurrent if gpus == 1 else 1
        cmd = ["sbatch", "--parsable", f"--array=0-{len(gl) - 1}%{conc}", f"--gres=gpu:{gpus}",
               f"--export=ALL,GROUPS={gfile.relative_to(ROOT)},MANIFEST={mpath.relative_to(ROOT)}"]
        if after:
            cmd.append(f"--dependency={after}" if ":" in str(after) else f"--dependency=afterok:{after}")
        site = os.environ.get("ICML_SITE", "euler")
        cmd.append("icml2027/slurm/hse/cell_group.sbatch" if site == "hse" else "icml2027/slurm/cell_group.sbatch")
        print(" ".join(cmd))
        if not dry:
            jid = subprocess.check_output(cmd, text=True, cwd=ROOT).strip()
            jobs.append({"label": label, "job_id": jid, "groups": len(gl)})
            print("submitted", jid, flush=True)
    stf = QDIR / "state.json"
    state = json.loads(stf.read_text()) if stf.exists() else {"jobs": []}
    state["jobs"] += [{**j, "grid": man["name"]} for j in jobs]
    state["counts"] = count_states()
    stf.write_text(json.dumps(state, indent=1))
    return jobs


def status():
    per = collections.defaultdict(collections.Counter)
    for f in CELLS.glob("*/status.json"):
        per[f.parent.name.split("__")[1]][json.loads(f.read_text())["state"]] += 1
    states = sorted({s for c in per.values() for s in c})
    print("| model | " + " | ".join(states) + " |")
    print("|---" * (len(states) + 1) + "|")
    for m, c in sorted(per.items()):
        print(f"| {m} | " + " | ".join(str(c.get(s, 0)) for s in states) + " |")
    total = count_states()
    print("total:", total)
    return total


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--grid", required=True)
    s.add_argument("--max-concurrent", type=int, default=3)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--after", default=None)
    s.add_argument("--reset-running", action="store_true", help="treat cells stuck in 'running' (dead job) as pending")
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "submit":
        submit(a.grid, a.max_concurrent, a.dry_run, a.after, a.reset_running)
    else:
        status()


if __name__ == "__main__":
    main()
