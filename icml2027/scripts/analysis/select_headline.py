#!/usr/bin/env python
"""Pre-registered headline rule (PREREGISTRATION section 2): after Tier-1, the arm with the largest mean
improvement in calibrated RPS (most negative delta) over all Tier-1 cells, on the two models where that arm's
mean improvement is largest, gets seeds 4 and 5 on every cluster. Writes icml2027/configs/grid_headline.yaml.
Baseline arms (random, paraphrase, bestofb) are eligible only if they win -- the rule is applied as written."""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "icml2027/results"))
    ap.add_argument("--out", default=str(ROOT / "icml2027/configs/grid_headline.yaml"))
    a = ap.parse_args()
    rows = []
    for d in Path(a.results, "cells").glob("*"):
        f = d / "eval_frozen.json"
        if not f.exists() or not (d / "config.json").exists():
            continue
        conf = json.loads((d / "config.json").read_text())
        if conf.get("fitness", "rps_cal") != "rps_cal" or conf.get("mutator", "Qwen/Qwen3.8-27B") != "Qwen/Qwen3.8-27B":
            continue
        if conf["seed"] not in (1, 2, 3) or conf["arm"] == "nsga2":
            continue
        ev = json.loads(f.read_text())
        if ev.get("excluded"):
            continue
        rows.append({"arm": conf["arm"], "model": conf["model_slug"], "delta": ev["delta"]["rps_cal"]})
    t = pd.DataFrame(rows)
    by_arm = t.groupby("arm")["delta"].mean().sort_values()
    best = by_arm.index[0]
    models = t[t.arm == best].groupby("model")["delta"].mean().sort_values().index[:2].tolist()
    grid = {"name": "headline", "budget_B": 80, "fitness": "rps_cal", "arms": [best], "models_tier": [1],
            "models": models, "clusters": [0, 1, 2, 3], "seeds": [4, 5],
            "selection": {"rule": "largest mean -delta RPS_cal over Tier-1", "arm_means": by_arm.round(5).to_dict(),
                          "n_cells": int(len(t))}}
    Path(a.out).write_text(yaml.safe_dump(grid, sort_keys=False))
    print("headline:", best, models)


if __name__ == "__main__":
    main()
