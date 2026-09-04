"""
E2b -- Head-to-head: non-LLM baselines vs the paper's own runs, on the paper's
own train/test case IDs.

For every committed run under results_experiments/evoprompt_iter2/, this reads
cluster_<k>/{train,test}_case_ids.csv, fits the trivial baselines on exactly
those train cases, scores them on exactly those test cases, and puts the numbers
next to that run's before/after `mean_similarity` from result_log.json.

No new LLM calls. Pure arithmetic on data already in the repo.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ITEMS = [f"i{i}" for i in range(1, 121)]
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "arr2026/results/e2b"


def sim(pred, human):
    return float(np.mean(1.0 - np.abs(np.asarray(pred, float) - np.asarray(human, float)) / 4.0))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")
    by_case = df.set_index("case")
    grand = np.round(np.full(120, df[ITEMS].to_numpy(float).mean()))
    rng = np.random.default_rng(2026)

    rows = []
    for log in sorted((ROOT / "results_experiments/evoprompt_iter2").rglob("result_log.json")):
        run_dir = log.parent
        try:
            res = json.load(open(log))
        except Exception:
            continue
        model = run_dir.parent.name
        for cdir in sorted(run_dir.glob("cluster_*")):
            cid = cdir.name.split("_")[1]
            tr_f, te_f = cdir / "train_case_ids.csv", cdir / "test_case_ids.csv"
            if not (tr_f.exists() and te_f.exists()):
                continue
            tr_cases = pd.read_csv(tr_f)["case"].tolist()
            te_cases = pd.read_csv(te_f)["case"].tolist()
            tr = by_case.loc[[c for c in tr_cases if c in by_case.index]]
            te = by_case.loc[[c for c in te_cases if c in by_case.index]]
            if len(tr) == 0 or len(te) == 0:
                continue
            tri, tei = tr[ITEMS].to_numpy(float), te[ITEMS].to_numpy(float)

            majority = np.array([pd.Series(tri[:, j]).mode().iloc[0] for j in range(120)])
            meanans = np.round(tri.mean(axis=0))
            base = {
                "cluster_mean": meanans,
                "cluster_majority": majority,
                "global_mean": grand,
            }
            scores = {k: float(np.mean([sim(v, tei[i]) for i in range(len(tei))]))
                      for k, v in base.items()}
            # stochastic baselines: average over 20 draws
            emp, cpy = [], []
            for _ in range(20):
                e = np.array([rng.choice(tri[:, j]) for j in range(120)])
                emp.append(np.mean([sim(e, tei[i]) for i in range(len(tei))]))
                c = tri[rng.integers(len(tri))]
                cpy.append(np.mean([sim(c, tei[i]) for i in range(len(tei))]))
            scores["empirical_sample"] = float(np.mean(emp))
            scores["train_user_copy"] = float(np.mean(cpy))

            st = (res.get("clusters", {}).get(cid, {}) or {}).get("stages", {})
            before = (st.get("before_optimization_test", {}).get("summary", {}) or {}).get("mean_similarity")
            after = (st.get("after_optimization_test", {}).get("summary", {}) or {}).get("mean_similarity")

            rows.append({
                "model": model, "cluster": int(cid), "run": run_dir.name,
                "n_train": len(tr), "n_test": len(te),
                "llm_before": before, "llm_after": after,
                **{f"base_{k}": round(v, 4) for k, v in scores.items()},
                "best_baseline": max(scores, key=scores.get),
                "best_baseline_score": round(max(scores.values()), 4),
                "llm_beats_best_baseline": (
                    None if after is None else bool(after > max(scores.values()))),
            })

    t = pd.DataFrame(rows).sort_values(["model", "cluster"])
    t.to_csv(OUT / "head_to_head.csv", index=False)

    done = t[t.llm_after.notna()]
    summary = {
        "n_cells": int(len(done)),
        "n_cells_llm_beats_best_baseline": int(done.llm_beats_best_baseline.sum()),
        "mean_llm_after": round(float(done.llm_after.mean()), 4),
        "mean_best_baseline": round(float(done.best_baseline_score.mean()), 4),
        "mean_gap_baseline_minus_llm": round(
            float((done.best_baseline_score - done.llm_after).mean()), 4),
        "mean_cluster_mean_baseline": round(float(done.base_cluster_mean.mean()), 4),
        "mean_global_mean_baseline": round(float(done.base_global_mean.mean()), 4),
        "per_model": {
            m: {"llm_after": round(float(g.llm_after.mean()), 4),
                "best_baseline": round(float(g.best_baseline_score.mean()), 4),
                "cells_llm_wins": int(g.llm_beats_best_baseline.sum()),
                "cells": int(len(g))}
            for m, g in done.groupby("model")},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    cols = ["model", "cluster", "llm_before", "llm_after", "base_cluster_mean",
            "base_cluster_majority", "base_global_mean", "llm_beats_best_baseline"]
    print(done[cols].to_string(index=False))
    print("\n=== E2b SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
