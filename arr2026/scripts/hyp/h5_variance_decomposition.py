"""
H5 -- "43% of human variance" is the wrong summary.

Decompose each answer matrix as  Y_pj = mu + a_p + b_j + (ab)_pj + e  where
  a_p     respondent main effect  (this person answers high/low overall)
  b_j     item main effect        (this item is endorsed by everyone / no one)
  (ab)_pj person-by-item interaction -- the INDIVIDUATING component, the thing
          a personality simulator is actually supposed to reproduce.

A single variance ratio collapses these three into one number, and they carry
opposite diagnoses. If the simulator's shortfall is concentrated in (ab)_pj while
its item main effect is at or above the human level, then it is not "emitting too
little variance" -- it is emitting roughly the right total in the wrong place: a
confident cluster-level answer key, re-used for every respondent.

Reports the interaction variance ratio (IVR) beside the plain VR, each with a
constant floor (IVR = 0 by construction) and a human ceiling.

CPU only. No LLM calls.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ITEMS = [f"i{i}" for i in range(1, 121)]
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "arr2026/results/h5"


def decompose(Y):
    """Two-way crossed decomposition. Returns variance shares summing to ~1."""
    Y = np.asarray(Y, float)
    grand = np.nanmean(Y)
    a = np.nanmean(Y, axis=1) - grand           # respondent main effect
    b = np.nanmean(Y, axis=0) - grand           # item main effect
    resid = Y - (grand + a[:, None] + b[None, :])
    va, vb, vr = np.nanvar(a), np.nanvar(b), np.nanvar(resid)
    tot = va + vb + vr
    return {"var_total": float(tot), "var_person": float(va),
            "var_item": float(vb), "var_interaction": float(vr),
            "share_person": float(va / tot), "share_item": float(vb / tot),
            "share_interaction": float(vr / tot),
            "sd_interaction": float(np.sqrt(vr))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")
    rng = np.random.default_rng(2026)
    rows = []

    for cdir in sorted((ROOT / "results_experiments/evoprompt_iter2").glob("*/*/cluster_*")):
        cid = int(cdir.name.split("_")[1]); model = cdir.parent.parent.name
        f = cdir / "after_optimization_test_answers.csv"
        tr_f, te_f = cdir / "train_case_ids.csv", cdir / "test_case_ids.csv"
        if not (f.exists() and te_f.exists() and tr_f.exists()):
            continue
        M = pd.read_csv(f)[ITEMS].to_numpy(float)
        if np.isnan(M).mean() > 0.5 or len(np.unique(M[~np.isnan(M)])) < 2:
            continue
        te = set(pd.read_csv(te_f)["case"]); tr = set(pd.read_csv(tr_f)["case"])
        cl = df[df.clusters == cid]
        H = cl[cl.case.isin(te)][ITEMS].to_numpy(float)
        pool = cl[~cl.case.isin(te | tr)][ITEMS].to_numpy(float)
        tri = cl[cl.case.isin(tr)][ITEMS].to_numpy(float)
        if len(H) < 8 or len(pool) < len(H) or len(tri) == 0:
            continue
        Hc = pool[rng.choice(len(pool), len(H), replace=False)]      # human ceiling
        C = np.tile(np.round(tri.mean(axis=0)), (len(H), 1))          # constant floor

        dm, dh, dc = decompose(M), decompose(Hc), decompose(C)
        rows.append({
            "model": model, "cluster": cid,
            "model_var_total": dm["var_total"], "human_var_total": dh["var_total"],
            "model_share_item": dm["share_item"], "human_share_item": dh["share_item"],
            "model_share_interaction": dm["share_interaction"],
            "human_share_interaction": dh["share_interaction"],
            "model_share_person": dm["share_person"], "human_share_person": dh["share_person"],
            "total_var_ratio": dm["var_total"] / dh["var_total"],
            "IVR_model": dm["sd_interaction"] / dh["sd_interaction"],
            "IVR_constant": dc["sd_interaction"] / dh["sd_interaction"],
        })
        print(f"  {model:13s} c{cid} totVar {dm['var_total']:.2f} vs {dh['var_total']:.2f} "
              f"| item {dm['share_item']:.0%} vs {dh['share_item']:.0%} "
              f"| inter {dm['share_interaction']:.0%} vs {dh['share_interaction']:.0%} "
              f"| IVR={rows[-1]['IVR_model']:.3f}", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(OUT / "variance_components.csv", index=False)
    summary = {
        "n_cells": int(len(t)),
        "total_variance_ratio_mean": round(float(t.total_var_ratio.mean()), 4),
        "share_item_model_mean": round(float(t.model_share_item.mean()), 4),
        "share_item_human_mean": round(float(t.human_share_item.mean()), 4),
        "share_interaction_model_mean": round(float(t.model_share_interaction.mean()), 4),
        "share_interaction_human_mean": round(float(t.human_share_interaction.mean()), 4),
        "IVR_model_mean": round(float(t.IVR_model.mean()), 4),
        "IVR_constant_mean": round(float(t.IVR_constant.mean()), 4),
        "interpretation": (
            "If total_variance_ratio is near or above 1 while IVR is far below 1, the "
            "simulator emits roughly the right amount of variance in the wrong component: "
            "a cluster-level answer key rather than individuating behaviour."),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== H5 SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
