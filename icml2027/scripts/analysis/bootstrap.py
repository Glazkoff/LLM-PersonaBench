#!/usr/bin/env python
"""Paired bootstrap of every arm's evolved persona against base / random / paraphrase / best-of-B on the
same frozen panel, BH across the whole family, and a per-arm summary."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def boot(d, B, seed):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    n = len(d)
    bs = np.array([np.nanmean(d[rng.integers(0, n, n)]) for _ in range(B)])
    p = 2 * min(np.mean(bs >= 0), np.mean(bs <= 0))
    return float(np.nanmean(d)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), max(p, 1.0 / B)


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return p
    o = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank in range(m, 0, -1):
        i = o[rank - 1]
        prev = min(prev, p[i] * m / rank)
        q[i] = prev
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--panels", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.panels))
    B, seed = cfg["bootstrap"]["B"], cfg["bootstrap"]["seed"]
    R, O = Path(a.results), Path(a.out)
    per = {}
    for d in (R / "cells").glob("*"):
        if (d / "per_respondent.json").exists() and (d / "eval_frozen.json").exists():
            if json.loads((d / "eval_frozen.json").read_text()).get("excluded"):
                continue
            arm, slug, c, s = d.name.split("__")
            per[(arm, slug, c, s)] = json.loads((d / "per_respondent.json").read_text())
    rows = []
    for (arm, slug, c, s), v in sorted(per.items()):
        ev = np.array(v["evolved_rps_cal"])
        comps = {"base": np.array(v["base_rps_cal"])}
        for comp in ("random", "paraphrase", "bestofb"):
            if comp != arm and (comp, slug, c, s) in per:
                comps[comp] = np.array(per[(comp, slug, c, s)]["evolved_rps_cal"])
        for comp, x in comps.items():
            m, lo, hi, p = boot(ev - x, B, seed)
            rows.append({"arm": arm, "model": slug, "cluster": c, "seed": s, "comparator": comp,
                         "delta": m, "lo": lo, "hi": hi, "p": p})
    t = pd.DataFrame(rows)
    if len(t):
        t["q_bh"] = bh(t["p"].values)
    t.to_csv(O / "tests_bh.csv", index=False)
    summ = []
    for arm, g in (t.groupby("arm") if len(t) else []):
        vb = g[g.comparator == "base"]
        d = vb["delta"].to_numpy()
        if len(d) > 1:
            m, lo, hi, _ = boot(d, B, seed)
        else:
            m, lo, hi = float(np.mean(d)) if len(d) else np.nan, np.nan, np.nan
        row = {"arm": arm, "cells": len(vb), "mean_delta_vs_base": m, "lo": lo, "hi": hi}
        for k in ("base", "random", "paraphrase", "bestofb"):
            gk = g[g.comparator == k]
            row[f"n_vs_{k}"] = len(gk)
            row[f"sig_better_vs_{k}"] = int(((gk.q_bh < cfg["bh_q"]) & (gk.delta < 0)).sum())
            row[f"sig_worse_vs_{k}"] = int(((gk.q_bh < cfg["bh_q"]) & (gk.delta > 0)).sum())
        summ.append(row)
    pd.DataFrame(summ).to_csv(O / "tier1_summary.csv", index=False)
    print(len(t), "tests;", len(summ), "arms")


if __name__ == "__main__":
    main()
