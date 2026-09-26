#!/usr/bin/env python
"""One row per completed cell, plus evaluations-to-threshold, from icml2027/results/cells."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R, O = Path(a.results), Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    rows, e2t = [], []
    for d in sorted((R / "cells").glob("*")):
        f = d / "eval_frozen.json"
        if not f.exists():
            continue
        ev = json.loads(f.read_text())
        meta = json.loads((d / "run_meta.json").read_text())
        ms = json.loads((d / "mstar.json").read_text()) if (d / "mstar.json").exists() else {}
        bud = json.loads((d / "budget.json").read_text()) if (d / "budget.json").exists() else {}
        rows.append({"cell_id": d.name, "arm": meta["arm"], "fitness": meta.get("fitness", "rps_cal"),
                     "model": meta["model_slug"], "cluster": meta["cluster"], "seed": meta["seed"],
                     "base_rps_cal": ev["base"]["rps_cal"], "evolved_rps_cal": ev["evolved"]["rps_cal"],
                     "delta": ev["delta"]["rps_cal"], "ci_lo": ev["delta"]["rps_cal_ci95"][0],
                     "ci_hi": ev["delta"]["rps_cal_ci95"][1],
                     "base_rps_raw": ev["base"]["rps_raw"], "evolved_rps_raw": ev["evolved"]["rps_raw"],
                     "base_s0": ev["base"]["s0"], "evolved_s0": ev["evolved"]["s0"],
                     "alpha_base": ev["base"]["alpha"], "alpha_evolved": ev["evolved"]["alpha"],
                     "vr_between_base": ev["base"]["vr_between"], "vr_between_evolved": ev["evolved"]["vr_between"],
                     "wrong_persona": ev["evolved"].get("wrong_persona_rps_cal"),
                     "mass": ev["evolved"]["mass_on_scale"], "excluded": ev["excluded"],
                     "mstar_base": ms.get("base_interp"), "mstar_evolved": ms.get("evolved_interp"),
                     "mstar_delta": ms.get("delta_interp"), "prompt_tokens": bud.get("prompt_tokens"),
                     "generated_tokens": bud.get("generated_tokens"),
                     **{f"floor_{k}": v for k, v in ev["floors"].items()}})
        base = ev["base"]["rps_cal"]
        hist = [json.loads(l) for l in (d / "history.jsonl").read_text().splitlines() if l.strip()]
        best = np.minimum.accumulate([h["rps_cal_opt"] for h in hist]) if hist else []
        init = hist[0]["rps_cal_opt"] if hist else np.nan
        hit = next((h["eval_idx"] for h, b in zip(hist, best) if b <= init - 0.005), np.nan)
        e2t.append({"cell_id": d.name, "arm": meta["arm"], "model": meta["model_slug"], "evals": hit,
                    "final_best_opt": float(best[-1]) if len(best) else np.nan, "init_opt": init})
    pd.DataFrame(rows).to_csv(O / "tier1_cells.csv", index=False)
    pd.DataFrame(e2t).to_csv(O / "evaluations_to_threshold.csv", index=False)
    print(len(rows), "cells aggregated")


if __name__ == "__main__":
    main()
