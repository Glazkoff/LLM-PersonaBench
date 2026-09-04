"""
H7 -- The wrong-persona control that this literature never runs.

No persona-simulation paper we have found reports what happens when a simulator
conditioned on cluster c is scored against the humans of a DIFFERENT cluster c'.
Without that control, no reported persona effect is attributable to the persona:
it could be a generic survey-respondent effect that any prompt would produce.

Builds the full 4x4 matrix of (persona cluster) x (human cluster) for every
metric -- the paper's S, and our C2ST / DF / VR -- and tests diagonal dominance
by permutation. If the diagonal does not beat the off-diagonal, cluster
conditioning carries no measurable signal.

Runs entirely on committed answers. No LLM calls.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ITEMS = [f"i{i}" for i in range(1, 121)]
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "arr2026/results/h7"
SEED, N_REF = 2026, 40


def sim(pred, human):
    n = min(len(pred), len(human))
    return float(np.nanmean([np.nanmean(1 - np.abs(pred[i] - human[i]) / 4) for i in range(n)]))


def df_metric(model, human):
    ds = []
    for j in range(model.shape[1]):
        a = model[:, j][~np.isnan(model[:, j])]; b = human[:, j][~np.isnan(human[:, j])]
        if len(a) and len(b):
            ds.append(wasserstein_distance(a, b))
    return float(1 - np.mean(ds) / 4)


def c2st(a, b, seed=SEED):
    if len(a) < 8 or len(b) < 8:
        return float("nan")
    X = np.vstack([a, b]); y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    aucs = []
    for tr, te in StratifiedKFold(4, shuffle=True, random_state=seed).split(X, y):
        if len(np.unique(y[te])) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=120, random_state=seed).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(max(np.mean(aucs), 1 - np.mean(aucs))) if aucs else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")
    rng = np.random.default_rng(SEED)

    # model answers per (model, persona cluster)
    answers, used = {}, {}
    for cdir in sorted((ROOT / "results_experiments/evoprompt_iter2").glob("*/*/cluster_*")):
        cid = int(cdir.name.split("_")[1]); m = cdir.parent.parent.name
        f = cdir / "after_optimization_test_answers.csv"
        tr_f, te_f = cdir / "train_case_ids.csv", cdir / "test_case_ids.csv"
        if not (f.exists() and te_f.exists() and tr_f.exists()):
            continue
        M = pd.read_csv(f)[ITEMS].to_numpy(float)
        if np.isnan(M).mean() > 0.5 or len(np.unique(M[~np.isnan(M)])) < 2:
            continue
        answers[(m, cid)] = M
        used[(m, cid)] = set(pd.read_csv(te_f)["case"]) | set(pd.read_csv(tr_f)["case"])

    clusters = sorted({c for _, c in answers})
    rows = []
    for (m, pc), M in answers.items():
        for hc in clusters:
            cl = df[df.clusters == hc]
            pool = cl[~cl.case.isin(used.get((m, hc), set()))][ITEMS].to_numpy(float)
            if len(pool) < N_REF:
                continue
            ref = pool[rng.choice(len(pool), N_REF, replace=False)]
            ok = np.nanstd(ref, 0) > 0
            rows.append({
                "model": m, "persona_cluster": pc, "human_cluster": hc,
                "diagonal": pc == hc,
                "S": sim(M, ref), "DF": df_metric(M, ref), "C2ST": c2st(M, ref),
                "VR": float(np.nanmean(np.nanstd(M, 0)[ok] / np.nanstd(ref, 0)[ok])),
            })
        print(f"  {m:13s} persona c{pc} scored against all {len(clusters)} human clusters",
              flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(OUT / "cross_matrix.csv", index=False)

    summary = {"n_rows": int(len(t)), "metrics": {}}
    for met, better_high in (("S", True), ("DF", True), ("C2ST", False), ("VR", True)):
        d = t[t.diagonal][met].mean(); o = t[~t.diagonal][met].mean()
        gap = (d - o) if better_high else (o - d)
        # permutation test on the diagonal advantage
        obs, cnt, B = gap, 0, 2000
        lab = t.diagonal.to_numpy()
        vals = t[met].to_numpy()
        for _ in range(B):
            p = rng.permutation(lab)
            g = (np.nanmean(vals[p]) - np.nanmean(vals[~p])) * (1 if better_high else -1)
            if g >= obs:
                cnt += 1
        summary["metrics"][met] = {
            "diagonal_mean": round(float(d), 4), "offdiagonal_mean": round(float(o), 4),
            "advantage": round(float(gap), 4), "perm_p": round((cnt + 1) / (B + 1), 4)}
        print(f"  {met}: diag={d:.4f} offdiag={o:.4f} advantage={gap:+.4f} "
              f"p={(cnt+1)/(B+1):.4f}", flush=True)

    summary["interpretation"] = (
        "A diagonal advantage that survives permutation is the first evidence that "
        "cluster conditioning carries signal at all. A null makes the wrong-persona "
        "control a mandatory baseline, exactly as the constant floor is.")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== H7 SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
