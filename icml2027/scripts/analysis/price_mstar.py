#!/usr/bin/env python
"""Answer-equivalent m* of the base and evolved personas of every completed cell.

Uses the audit's own decoder (arr2026/scripts/hyp/h26_answer_equivalent.fit_decoder: ridge on one-hot observed
answers onto threshold indicators, penalty retuned at EVERY budget on the calibration respondents, isotonic
projection). Observed answers are the persona's 60 input items; targets are the 60 scored items; the decoder
is fit on the cell's training block and selected on its calibration panel, and scored on the frozen eval panel,
i.e. on exactly the respondents behind eval_frozen.json. Curves are cached per (cluster, seed)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "arr2026/scripts/hyp"))
from h25_incremental_screen import onehot, rps as rps_cdf  # noqa: E402
from h26_answer_equivalent import BUDGETS, fit_decoder  # noqa: E402

from src.optim.panels import ITEM_SPLIT, answers, excluded_cases, load_corpus, make_panels  # noqa: E402
from src.optim.results import atomic_write_json  # noqa: E402

K = 5


def decoder_curve(panels, reps, seed):
    inp, tgt = ITEM_SPLIT
    Xtr, Ytr = answers(panels.train, inp), answers(panels.train, tgt)
    Xs = np.vstack([answers(panels.cal, inp), answers(panels.eval, inp)])
    Yt = np.vstack([answers(panels.cal, tgt), answers(panels.eval, tgt)])
    n, nt = Yt.shape
    CAL, EV = np.arange(len(panels.cal)), np.arange(len(panels.cal), n)
    prior_row = np.stack([np.nanmean(Ytr <= t, axis=0) for t in range(1, K)], axis=1)
    targ = np.nan_to_num(np.concatenate([(Ytr <= t).astype(float) for t in range(1, K)], axis=1), nan=0.0)
    rng = np.random.default_rng(seed)
    curve = {}
    for m in [b for b in BUDGETS if b <= len(inp)]:
        vals = []
        for _ in range(1 if m in (0, len(inp)) else reps):
            if m == 0:
                D = np.broadcast_to(prior_row[None], (n, nt, K - 1))
            else:
                cols = rng.choice(len(inp), m, replace=False)
                D, _ = fit_decoder(onehot(Xtr[:, cols], K), targ, onehot(Xs[:, cols], K), n, nt, K, None, Yt, CAL)
            vals.append(float(np.nanmean(rps_cdf(D[EV], Yt[EV], K))))
        curve[m] = float(np.mean(vals))
    return curve


def interp_mstar(curve, score):
    ms = sorted(curve)
    if score >= curve[ms[0]]:
        return 0.0
    for lo, hi in zip(ms, ms[1:]):
        if curve[lo] >= score >= curve[hi] and curve[lo] > curve[hi]:
            return float(lo + (curve[lo] - score) / (curve[lo] - curve[hi]) * (hi - lo))
    return float(ms[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--panels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.panels))
    df = load_corpus()
    ex = excluded_cases(cfg["exclude_case_ids_from"])
    cdir = Path(a.out) / "decoder_curves"
    cdir.mkdir(parents=True, exist_ok=True)
    curves, rows = {}, []
    for d in sorted(Path(a.results, "cells").glob("*")):
        f = d / "eval_frozen.json"
        if not f.exists():
            continue
        ev = json.loads(f.read_text())
        c, s = ev["panel"]["cluster"], ev["panel"]["seed"]
        if (c, s) not in curves:
            cf = cdir / f"c{c}__s{s}.json"
            if cf.exists():
                curves[(c, s)] = {int(k): v for k, v in json.loads(cf.read_text()).items()}
            else:
                curves[(c, s)] = decoder_curve(make_panels(df, c, s, cfg["panel_sizes"], ex), a.reps, 20260918 + s)
                atomic_write_json(cf, curves[(c, s)])
                print("curve", c, s, curves[(c, s)], flush=True)
        cur = curves[(c, s)]
        mb, me = interp_mstar(cur, ev["base"]["rps_cal"]), interp_mstar(cur, ev["evolved"]["rps_cal"])
        atomic_write_json(d / "mstar.json", {"curve": cur, "base_interp": mb, "evolved_interp": me,
                                             "delta_interp": me - mb})
        arm, slug, _, _ = d.name.split("__")
        rows.append({"cell_id": d.name, "arm": arm, "model": slug, "cluster": c, "seed": s,
                     "mstar_base": mb, "mstar_evolved": me, "delta": me - mb})
    pd.DataFrame(rows).to_csv(Path(a.out) / "mstar_by_arm_model.csv", index=False)
    print(len(rows), "cells priced")


if __name__ == "__main__":
    main()
