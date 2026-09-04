"""
E3 -- A replacement evaluation suite for cluster-level personality simulation.

WHY THE OLD METRIC FAILS (proved in e3_proposition.py, measured here)
--------------------------------------------------------------------
The paper scores a simulated respondent x against a human y with
    S(x, y) = mean_j [ 1 - |x_j - y_j| / 4 ].
Averaged over humans y drawn from a cluster's per-item response distribution P_j,
the best possible *constant* predictor is the per-item median m_j, scoring
    1 - E_Y|m_j - Y| / 4          (mean absolute deviation about the median)
while a predictor that faithfully SAMPLES from P_j scores
    1 - E_{X,Y}|X - Y| / 4        (Gini mean difference)
and E|X-Y| >= min_c E|c-Y| = E|m-Y| always. So a perfectly faithful simulator is
provably dominated by a constant, strictly so for any non-degenerate item. The
metric cannot reward distributional fidelity; it rewards central tendency.

THE SUITE
---------
Every metric below is reported together with its CONSTANT FLOOR and its HUMAN
CEILING, so a number is only meaningful relative to what a constant achieves and
what a real cluster member achieves.

  C2ST-AUC   (primary) Classifier two-sample test. Can a classifier tell 40
             simulated respondents from 40 held-out humans of the same cluster?
             0.5 = indistinguishable (ideal twin). 1.0 = trivially separable.
             A constant scores ~1.0 by construction; a real human scores ~0.5.
  DF         Distributional fidelity: 1 - mean_j W1(model_j, human_j)/4, where
             W1 is the per-item Wasserstein-1 distance between the simulated and
             human response distributions. A constant is heavily penalised.
  VR         Variance ratio: mean_j sd(model_j)/sd(human_j). Constant -> 0,
             faithful -> 1. Pure diagnostic, not to be optimised directly.
  S_answer   The paper's original answer similarity, retained ONLY so it can be
             shown against its own floor and ceiling.

All of this runs off answers already committed in results_experiments/.
No new LLM calls.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ITEMS = [f"i{i}" for i in range(1, 121)]
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "arr2026/results/e3"
SEED = 2026
N_REF = 40          # held-out humans per comparison
N_REPEAT = 10       # repeats of the C2ST draw


# ---------------------------------------------------------------- metrics
def s_answer(model, human):
    """Paper metric: mean over paired respondents of mean item similarity."""
    n = min(len(model), len(human))
    return float(np.mean([
        np.nanmean(1.0 - np.abs(model[i] - human[i]) / 4.0) for i in range(n)]))


def distributional_fidelity(model, human):
    """1 - mean_j W1(model_j, human_j) / 4, per-item Wasserstein-1."""
    ds = []
    for j in range(model.shape[1]):
        a = model[:, j][~np.isnan(model[:, j])]
        b = human[:, j][~np.isnan(human[:, j])]
        if len(a) == 0 or len(b) == 0:
            continue
        ds.append(wasserstein_distance(a, b))
    return float(1.0 - np.mean(ds) / 4.0)


def variance_ratio(model, human):
    ms = np.nanstd(model, axis=0)
    hs = np.nanstd(human, axis=0)
    ok = hs > 1e-9
    return float(np.mean(ms[ok] / hs[ok]))


def c2st_auc(a, b, seed=SEED):
    """Classifier two-sample test. 0.5 = indistinguishable."""
    X = np.vstack([a, b])
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    if len(a) < 8 or len(b) < 8:
        return float("nan")
    aucs = []
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        if len(np.unique(y[te])) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=120, random_state=seed)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    if not aucs:
        return float("nan")
    # symmetric: distinguishability, direction-free
    return float(max(np.mean(aucs), 1 - np.mean(aucs)))


# ---------------------------------------------------------------- driver
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")
    rng = np.random.default_rng(SEED)

    rows = []
    for log in sorted((ROOT / "results_experiments/evoprompt_iter2").rglob("result_log.json")):
        run = log.parent
        model_name = run.parent.name
        for cdir in sorted(run.glob("cluster_*")):
            cid = int(cdir.name.split("_")[1])
            te_f = cdir / "test_case_ids.csv"
            tr_f = cdir / "train_case_ids.csv"
            if not te_f.exists():
                continue
            te_cases = set(pd.read_csv(te_f)["case"])
            tr_cases = set(pd.read_csv(tr_f)["case"]) if tr_f.exists() else set()

            cl = df[df.clusters == cid]
            human_test = cl[cl.case.isin(te_cases)][ITEMS].to_numpy(float)
            pool = cl[~cl.case.isin(te_cases | tr_cases)][ITEMS].to_numpy(float)
            if len(human_test) < 8 or len(pool) < 2 * N_REF:
                continue

            # constant floor: per-item cluster mean from TRAIN, repeated
            tr_items = cl[cl.case.isin(tr_cases)][ITEMS].to_numpy(float)
            const_vec = np.round(tr_items.mean(axis=0)) if len(tr_items) else np.round(pool.mean(axis=0))
            const_block = np.tile(const_vec, (N_REF, 1))

            for stage in ("before", "after"):
                f = cdir / f"{stage}_optimization_test_answers.csv"
                if not f.exists():
                    continue
                m = pd.read_csv(f)
                model_ans = m[ITEMS].to_numpy(float)
                # skip degenerate/failed runs: e.g. gpt4_nano_cluster_1_2/cluster_3
                # is committed with 100% NaN answers and must not enter any average
                nan_frac = float(np.isnan(model_ans).mean())
                if nan_frac > 0.5 or len(np.unique(model_ans[~np.isnan(model_ans)])) < 2:
                    print(f"  SKIP degenerate {model_name} c{cid} {stage} "
                          f"(nan_frac={nan_frac:.2f}) -> {f}", flush=True)
                    continue

                aucs_model, aucs_human, aucs_const = [], [], []
                dfid_h, vr_h = [], []
                for r in range(N_REPEAT):
                    rr = np.random.default_rng(SEED + r)
                    idx = rr.choice(len(pool), 2 * N_REF, replace=False)
                    ref_a, ref_b = pool[idx[:N_REF]], pool[idx[N_REF:]]
                    aucs_model.append(c2st_auc(model_ans, ref_a, SEED + r))
                    aucs_human.append(c2st_auc(ref_b, ref_a, SEED + r))
                    aucs_const.append(c2st_auc(const_block, ref_a, SEED + r))
                    dfid_h.append(distributional_fidelity(ref_b, ref_a))
                    vr_h.append(variance_ratio(ref_b, ref_a))

                ref = pool[rng.choice(len(pool), N_REF, replace=False)]
                rows.append({
                    "model": model_name, "cluster": cid, "stage": stage,
                    "n_model": len(model_ans),
                    # --- primary
                    "c2st_model": float(np.nanmean(aucs_model)),
                    "c2st_human_ceiling": float(np.nanmean(aucs_human)),
                    "c2st_constant_floor": float(np.nanmean(aucs_const)),
                    # --- distributional
                    "df_model": distributional_fidelity(model_ans, ref),
                    "df_human_ceiling": float(np.mean(dfid_h)),
                    "df_constant_floor": distributional_fidelity(const_block, ref),
                    # --- variance
                    "vr_model": variance_ratio(model_ans, ref),
                    "vr_human_ceiling": float(np.mean(vr_h)),
                    "vr_constant_floor": variance_ratio(const_block, ref),
                    # --- old metric, for reference only
                    "s_answer_model": s_answer(model_ans, human_test),
                    "s_answer_human_ceiling": s_answer(
                        pool[rng.choice(len(pool), len(human_test), replace=False)], human_test),
                    "s_answer_constant_floor": s_answer(
                        np.tile(const_vec, (len(human_test), 1)), human_test),
                })
                print(f"  {model_name:13s} c{cid} {stage:6s} "
                      f"C2ST={rows[-1]['c2st_model']:.3f} "
                      f"(human {rows[-1]['c2st_human_ceiling']:.3f}, "
                      f"const {rows[-1]['c2st_constant_floor']:.3f})  "
                      f"VR={rows[-1]['vr_model']:.2f}", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(OUT / "metric_suite.csv", index=False)

    aft = t[t.stage == "after"]
    summary = {
        "n_cells": int(len(aft)),
        "C2ST_AUC (0.5=indistinguishable, lower is better)": {
            "model_mean": round(float(aft.c2st_model.mean()), 4),
            "human_ceiling_mean": round(float(aft.c2st_human_ceiling.mean()), 4),
            "constant_floor_mean": round(float(aft.c2st_constant_floor.mean()), 4),
        },
        "DF distributional fidelity (higher better)": {
            "model_mean": round(float(aft.df_model.mean()), 4),
            "human_ceiling_mean": round(float(aft.df_human_ceiling.mean()), 4),
            "constant_floor_mean": round(float(aft.df_constant_floor.mean()), 4),
        },
        "VR variance ratio (1.0 ideal)": {
            "model_mean": round(float(aft.vr_model.mean()), 4),
            "human_ceiling_mean": round(float(aft.vr_human_ceiling.mean()), 4),
            "constant_floor_mean": round(float(aft.vr_constant_floor.mean()), 4),
        },
        "S_answer (old metric; note constant floor EXCEEDS human ceiling)": {
            "model_mean": round(float(aft.s_answer_model.mean()), 4),
            "human_ceiling_mean": round(float(aft.s_answer_human_ceiling.mean()), 4),
            "constant_floor_mean": round(float(aft.s_answer_constant_floor.mean()), 4),
        },
        "separation_check": {
            "old_metric_ranks_constant_above_human": bool(
                aft.s_answer_constant_floor.mean() > aft.s_answer_human_ceiling.mean()),
            "C2ST_ranks_human_above_constant": bool(
                aft.c2st_human_ceiling.mean() < aft.c2st_constant_floor.mean()),
            "DF_ranks_human_above_constant": bool(
                aft.df_human_ceiling.mean() > aft.df_constant_floor.mean()),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== E3 SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
