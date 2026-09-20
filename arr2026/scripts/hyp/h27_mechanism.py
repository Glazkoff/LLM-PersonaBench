r"""H27: WHY is one persona worth more answers than another?

H26 prices every persona in the currency of the respondent's own answers and finds
a clear model ladder (BRIEF.md section 16.2). H27 asks what determines a model's
position on it, and connects the answer to the variance decomposition this paper
already contains.

Three candidate explanations, all measurable from artifacts that already exist:

  alpha*              how much of the LLM's person-specific deviation from its own
                      population mean survives validation selection. This is the
                      calibration coefficient H26 already fits.
  between-person share of the belief variance, and the variance ratio against
                      human spread -- both already computed per model by
                      h4_variance_ladder.py into variance_ladder.csv.
  raw accuracy        the uncalibrated loss.

The prediction registered before running: accuracy does NOT explain the ladder, and
dispersion does. A model whose beliefs barely move between people has nothing for
calibration to keep, so alpha* collapses to zero and the persona is worth zero
answers, however accurate its population-level forecast is. If that holds, the
paper's existing variance analysis stops being an audit finding and becomes the
mechanism that explains the method.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]


def canon(name: str) -> str:
    """Model ids appear as 'Qwen/Qwen3.6-35B-A3B' and as directory fragments like
    'h4n160_Qwen_Qwen3p6-35B-A3B'. Reduce both to one comparable key."""
    s = str(name).split("/")[-1]
    s = s.replace("p", ".") if re.match(r"^[A-Za-z]+\d+p\d", s) else s
    s = re.sub(r"[-_.]", "", s).lower()
    for suf in ("instruct", "it", "chat", "preview"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--equiv", nargs="+", required=True,
                    help="H26 json outputs")
    ap.add_argument("--ladders", default="arr2026/results_euler",
                    help="root holding */variance_ladder.csv")
    ap.add_argument("--runs-root", default="arr2026/results_euler",
                    help="root holding the run directories named in the H26 output")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for pat in a.equiv:
        for p in glob.glob(str(ROOT / pat)) or glob.glob(pat):
            rows += json.loads(Path(p).read_text())
    eq = pd.DataFrame(rows)
    if eq.empty:
        raise SystemExit("no H26 results found")
    eq["key"] = eq.model.map(canon)
    eq["mi"] = eq.answer_equivalent_interp

    # ---- non-circular dispersion, measured from the model's own output ------
    # alpha* is FITTED to the same loss m* is derived from, so "alpha* predicts
    # m*" is partly definitional and a reviewer will say so. The property alpha*
    # is a proxy for is how much the model's forecast actually moves between
    # people. Measure that directly from the saved belief tensor -- no fitting,
    # no labels -- and use it as the mechanism predictor.
    rroot = ROOT / a.runs_root if not Path(a.runs_root).is_absolute() else Path(a.runs_root)
    disp = []
    for name in eq.run.unique():
        f = rroot / name / "belief_probs.npy"
        if not f.exists():
            continue
        P = np.load(f)
        C = np.cumsum(P, axis=2)[:, :, :-1]                 # (n, items, K-1)
        between = np.nanmean(np.nanstd(C, axis=0))          # spread ACROSS people
        within = np.nanmean(np.nanstd(C, axis=1))           # spread across items
        ent = -np.nansum(np.where(P > 0, P * np.log(np.clip(P, 1e-12, None)), 0.0),
                         axis=2)
        disp.append(dict(run=name, belief_between_sd=float(between),
                         belief_within_sd=float(within),
                         belief_between_share=float(between / max(between + within, 1e-12)),
                         belief_entropy=float(np.nanmean(ent))))
    if disp:
        eq = eq.merge(pd.DataFrame(disp), on="run", how="left")
        print(f"measured belief dispersion directly for {len(disp)} of "
              f"{eq.run.nunique()} runs")

    lad = []
    root = ROOT / a.ladders if not Path(a.ladders).is_absolute() else Path(a.ladders)
    for f in sorted(root.glob("*/variance_ladder.csv")):
        try:
            t = pd.read_csv(f)
        except Exception:
            continue
        if "model" not in t.columns:
            continue
        g = t.groupby("model").mean(numeric_only=True).reset_index()
        g["src"] = f.parent.name
        lad.append(g)
    if not lad:
        raise SystemExit(f"no variance_ladder.csv under {root}")
    L = pd.concat(lad, ignore_index=True)
    L["key"] = L.model.map(canon)
    keep = [c for c in ("VR_belief_mixture", "share_between", "share_within",
                        "VR_readout", "belief_entropy_mean", "human_sd_mean")
            if c in L.columns]
    L = L.groupby("key")[keep].mean().reset_index()

    m = eq.merge(L, on="key", how="left")
    matched = m[m[keep[0]].notna()] if keep else m.iloc[:0]
    print(f"{len(eq)} priced cells; {matched.key.nunique()} models matched to a "
          f"variance ladder ({len(matched)} cells)\n")

    print(f"{'predictor':28s} {'n':>4s} {'spearman':>9s} {'p':>10s}")
    out = {}
    for col, lbl in ([("alpha", "alpha* (calibration, FITTED)"),
                      ("belief_between_sd", "between-person belief SD"),
                      ("belief_between_share", "between-person share"),
                      ("belief_entropy", "belief entropy"),
                      ("llm_raw", "raw LLM loss"),
                      ("llm_calibrated", "calibrated loss")] +
                     [(c, c) for c in keep]):
        sub = m[[col, "mi"]].dropna() if col in m.columns else pd.DataFrame()
        if len(sub) < 6:
            continue
        r, p = stats.spearmanr(sub[col], sub["mi"])
        out[col] = dict(n=int(len(sub)), rho=float(r), p=float(p))
        print(f"{lbl:28s} {len(sub):4d} {r:+9.3f} {p:10.2e}")

    # Does dispersion explain the ladder OVER AND ABOVE accuracy? Partial
    # correlation of m* with each dispersion measure, controlling for raw loss.
    print(f"\n{'partial | raw loss controlled':28s} {'n':>4s} {'spearman':>9s} {'p':>10s}")
    for col in keep + ["alpha", "belief_between_sd", "belief_between_share",
                       "belief_entropy"]:
        if col not in m.columns:
            continue
        sub = m[[col, "mi", "llm_raw"]].dropna()
        if len(sub) < 8:
            continue
        rk = sub.rank()
        rx = rk[col] - np.polyval(np.polyfit(rk.llm_raw, rk[col], 1), rk.llm_raw)
        ry = rk["mi"] - np.polyval(np.polyfit(rk.llm_raw, rk["mi"], 1), rk.llm_raw)
        r, p = stats.pearsonr(rx, ry)
        out[f"partial_{col}"] = dict(n=int(len(sub)), rho=float(r), p=float(p))
        print(f"{col:28s} {len(sub):4d} {r:+9.3f} {p:10.2e}")

    o = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(
        {"correlations": out,
         "cells": len(eq), "matched_cells": int(len(matched)),
         "matched_models": sorted(matched.key.unique().tolist())}, indent=2))
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
