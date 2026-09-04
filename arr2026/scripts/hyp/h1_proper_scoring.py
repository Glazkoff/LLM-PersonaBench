"""
H1 -- The audited metric is CRPS with its dispersion term deleted.

The continuous ranked probability score for a predictive distribution F and an
outcome y is
        CRPS(F, y) = E_F|X - y| - (1/2) E_F|X - X'|
and it is STRICTLY PROPER: it is uniquely optimised by F = P, the true
distribution. The metric this literature uses,
        S(x, y) = 1 - |x - y| / 4,
is the first term alone, with the dispersion reward deleted. Deleting it is
exactly what makes a constant optimal (our Proposition).

Define the one-parameter family
        S_lambda = 1 - [ E|X - y| - lambda * E|X - X'| ] / 4.
  lambda = 0    -> the audited metric. Improper. Constant-optimal.
  lambda = 1/2  -> CRPS. Strictly proper. Truth-optimal.
  lambda > 1/2  -> improper the other way; rewards over-dispersion.
lambda = 1/2 is therefore NOT a tunable knob -- it is the unique proper choice,
which pre-empts the "you tuned your own metric" objection.

This script:
  (a) verifies (1/2)GMD <= MAD on every item of every cluster -- the properness
      claim, and the mirror of the Proposition table;
  (b) re-scores every committed model cell under S_lambda for a sweep of lambda,
      and finds lambda*, the crossover at which the model overtakes the constant.
      lambda* is an interpretable measure of the dispersion deficit: the larger
      it is, the more dispersion the simulator is missing.

CPU only. No LLM calls.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ITEMS = [f"i{i}" for i in range(1, 121)]
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "arr2026/results/h1"
LAMBDAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.75, 1.0]


def gmd(v):
    """E|X-X'| for X,X' iid from the empirical distribution of v. O(n log n)."""
    x = np.sort(np.asarray(v, float))
    n = len(x)
    if n < 2:
        return 0.0
    i = np.arange(1, n + 1)
    return float(2.0 * np.sum((2 * i - n - 1) * x) / (n * n))


def s_lambda(model_mat, human_mat, lam):
    """
    S_lambda for a simulated POPULATION against a human population.
    E|X-y| is averaged over all (simulated, human) pairs per item;
    E|X-X'| is the simulated population's own Gini mean difference per item.
    """
    per_item = []
    for j in range(human_mat.shape[1]):
        m = model_mat[:, j][~np.isnan(model_mat[:, j])]
        h = human_mat[:, j][~np.isnan(human_mat[:, j])]
        if len(m) == 0 or len(h) == 0:
            continue
        cross = np.abs(m[:, None] - h[None, :]).mean()
        per_item.append(1.0 - (cross - lam * gmd(m)) / 4.0)
    return float(np.mean(per_item))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")
    by_case = df.set_index("case")

    # ---------- (a) properness check: (1/2)GMD <= MAD on every item ----------
    rows = []
    for c in sorted(df.clusters.unique()):
        sub = df[df.clusters == c][ITEMS].to_numpy(float)
        mads, gmds = [], []
        for j in range(120):
            v = sub[:, j][~np.isnan(sub[:, j])]
            mads.append(float(np.mean(np.abs(v - np.median(v)))))
            gmds.append(gmd(v))
        mads, gmds = np.array(mads), np.array(gmds)
        rows.append({
            "cluster": int(c), "n": int(len(sub)),
            "MAD": float(mads.mean()), "GMD": float(gmds.mean()),
            "half_GMD": float((gmds / 2).mean()),
            "items_half_GMD_le_MAD": int(((gmds / 2) <= mads + 1e-12).sum()),
            "items_GMD_ge_MAD": int((gmds >= mads - 1e-12).sum()),
        })
        print(f"cluster {c}: MAD={mads.mean():.4f} GMD={gmds.mean():.4f} "
              f"halfGMD={((gmds/2).mean()):.4f} | "
              f"halfGMD<=MAD on {rows[-1]['items_half_GMD_le_MAD']}/120, "
              f"GMD>=MAD on {rows[-1]['items_GMD_ge_MAD']}/120", flush=True)
    prop = pd.DataFrame(rows)
    prop.to_csv(OUT / "properness.csv", index=False)

    # ---------- (b) lambda sweep on the committed cells ----------
    sweep = []
    base = ROOT / "results_experiments/evoprompt_iter2"
    for cdir in sorted(base.glob("*/*/cluster_*")):
        cid = int(cdir.name.split("_")[1])
        model = cdir.parent.parent.name
        f = cdir / "after_optimization_test_answers.csv"
        tr_f, te_f = cdir / "train_case_ids.csv", cdir / "test_case_ids.csv"
        if not (f.exists() and tr_f.exists() and te_f.exists()):
            continue
        m = pd.read_csv(f)[ITEMS].to_numpy(float)
        if np.isnan(m).mean() > 0.5 or len(np.unique(m[~np.isnan(m)])) < 2:
            print(f"  SKIP degenerate {model} c{cid}", flush=True)
            continue
        te = pd.read_csv(te_f)["case"].tolist()
        tr = pd.read_csv(tr_f)["case"].tolist()
        human = by_case.loc[[c for c in te if c in by_case.index]][ITEMS].to_numpy(float)
        tri = by_case.loc[[c for c in tr if c in by_case.index]][ITEMS].to_numpy(float)
        const = np.tile(np.round(tri.mean(axis=0)), (len(human), 1))
        pool = df[(df.clusters == cid) & (~df.case.isin(set(te) | set(tr)))][ITEMS].to_numpy(float)
        rng = np.random.default_rng(2026)
        hum_sim = pool[rng.choice(len(pool), min(len(human), len(pool)), replace=False)]

        for lam in LAMBDAS:
            sweep.append({
                "model": model, "cluster": cid, "lambda": lam,
                "S_model": s_lambda(m, human, lam),
                "S_constant": s_lambda(const, human, lam),
                "S_human": s_lambda(hum_sim, human, lam),
            })
        print(f"  swept {model} c{cid}", flush=True)

    sw = pd.DataFrame(sweep)
    sw.to_csv(OUT / "lambda_sweep.csv", index=False)

    # crossover lambda* per cell: smallest lambda where model >= constant
    stars = []
    for (mo, cl), g in sw.groupby(["model", "cluster"]):
        g = g.sort_values("lambda")
        win = g[g.S_model >= g.S_constant]
        hwin = g[g.S_human >= g.S_constant]
        stars.append({
            "model": mo, "cluster": cl,
            "lambda_star_model": float(win["lambda"].iloc[0]) if len(win) else None,
            "lambda_star_human": float(hwin["lambda"].iloc[0]) if len(hwin) else None,
        })
    st = pd.DataFrame(stars)
    st.to_csv(OUT / "lambda_star.csv", index=False)

    agg = sw.groupby("lambda")[["S_model", "S_constant", "S_human"]].mean().round(4)
    summary = {
        "properness_half_GMD_le_MAD_all_items": bool(
            (prop.items_half_GMD_le_MAD == 120).all()),
        "improperness_GMD_ge_MAD_all_items": bool((prop.items_GMD_ge_MAD == 120).all()),
        "mean_over_cells_by_lambda": agg.to_dict("index"),
        "lambda_star_model_median": (None if st.lambda_star_model.isna().all()
                                     else float(st.lambda_star_model.median())),
        "lambda_star_human_median": (None if st.lambda_star_human.isna().all()
                                     else float(st.lambda_star_human.median())),
        "cells_where_human_never_beats_constant": int(st.lambda_star_human.isna().sum()),
        "cells_where_model_never_beats_constant": int(st.lambda_star_model.isna().sum()),
        "n_cells": int(len(st)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n=== H1 SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
