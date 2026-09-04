"""
Proposition (why the paper's metric cannot reward a faithful simulator) and its
numerical verification on the real IPIP-NEO-120 cluster response distributions.

SETUP. Fix a cluster and an item j. Let P_j be the cluster's response
distribution over {1..5}. The paper scores a simulated answer x against a human
answer Y ~ P_j with s(x, Y) = 1 - |x - Y| / 4.

PROPOSITION.
  (a) The optimal CONSTANT predictor is any median m_j of P_j, with expected
      score   1 - MAD_j / 4,   MAD_j = E_Y |m_j - Y|.
  (b) A FAITHFUL simulator, i.e. X ~ P_j independent of Y, has expected score
      1 - GMD_j / 4,   GMD_j = E_{X,Y} |X - Y|   (the Gini mean difference).
  (c) GMD_j >= MAD_j always, with equality iff P_j is a point mass. Hence
        E[s(constant)] >= E[s(faithful simulator)],
      strictly whenever the item has any response variance.

PROOF of (c). GMD_j = E_X [ E_Y |X - Y| ] >= E_X [ min_c E_Y |c - Y| ]
= E_Y |m_j - Y| = MAD_j, since the median minimises expected absolute deviation.
Equality forces X = m_j a.s., i.e. P_j degenerate.                        []

CONSEQUENCE. Under this metric the ideal target -- reproducing the cluster's
response distribution -- is strictly worse than emitting a constant. Optimising it
drives a simulator toward degenerate central-tendency responding, which is exactly
the answer-similarity-up / profile-fidelity-down pattern the paper reports and
files under future work.

This script computes MAD_j, GMD_j and the implied score gap on the real data.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ITEMS = [f"i{i}" for i in range(1, 121)]
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "arr2026/results/e3"


def mad_about_median(v):
    return float(np.mean(np.abs(v - np.median(v))))


def gini_mean_difference(v):
    """E|X-Y| for X,Y iid from the empirical distribution of v (exact, O(n log n))."""
    x = np.sort(np.asarray(v, float))
    n = len(x)
    i = np.arange(1, n + 1)
    # sum_{a,b} |x_a - x_b| = 2 * sum_i (2i - n - 1) x_i
    total = 2.0 * np.sum((2 * i - n - 1) * x)
    return float(total / (n * n))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")

    rows = []
    for c in sorted(df.clusters.unique()):
        sub = df[df.clusters == c][ITEMS].to_numpy(float)
        mads, gmds = [], []
        for j in range(120):
            v = sub[:, j]
            v = v[~np.isnan(v)]
            mads.append(mad_about_median(v))
            gmds.append(gini_mean_difference(v))
        mads, gmds = np.array(mads), np.array(gmds)
        rows.append({
            "cluster": int(c),
            "n": int(len(sub)),
            "MAD_mean": float(mads.mean()),
            "GMD_mean": float(gmds.mean()),
            "score_constant": float(1 - mads.mean() / 4),
            "score_faithful": float(1 - gmds.mean() / 4),
            "gap_pp": float((gmds.mean() - mads.mean()) / 4 * 100),
            "items_with_GMD_ge_MAD": int((gmds >= mads - 1e-12).sum()),
            "items_total": 120,
        })
        print(f"cluster {c}: constant={rows[-1]['score_constant']:.4f}  "
              f"faithful={rows[-1]['score_faithful']:.4f}  "
              f"gap={rows[-1]['gap_pp']:.2f} pp  "
              f"GMD>=MAD on {rows[-1]['items_with_GMD_ge_MAD']}/120 items", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(OUT / "proposition.csv", index=False)
    summary = {
        "proposition_holds_on_every_item": bool((t.items_with_GMD_ge_MAD == 120).all()),
        "mean_score_constant": round(float(t.score_constant.mean()), 4),
        "mean_score_faithful_simulator": round(float(t.score_faithful.mean()), 4),
        "mean_gap_pp": round(float(t.gap_pp.mean()), 3),
        "interpretation": (
            "A simulator that reproduces the cluster response distribution exactly "
            "is penalised by this many percentage points relative to a constant. "
            "Any reported improvement smaller than this gap is consistent with the "
            "model merely drifting toward the cluster mean."),
        "per_cluster": t.to_dict("records"),
    }
    (OUT / "proposition_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== PROPOSITION ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_cluster"}, indent=2))


if __name__ == "__main__":
    main()
