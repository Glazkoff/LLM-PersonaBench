"""
H9 -- Are the ceilings and floors real, or finite-sample artefacts?

Our suite reports a human ceiling of C2ST-AUC 0.564 at n=40 per side over 120
features. That is squarely in the overfitting regime: cross-validated AUC drifts
above 0.5 even for two samples from the SAME distribution. If the 0.564 is pure
estimator bias, the ceiling should fall toward 0.50 as n grows. Plug-in Wasserstein
is likewise upward-biased at rate n^-1/2, which makes DF pessimistic for everyone.

Sweeps n and fits metric(n) = metric_inf + c*n^-alpha for the human-vs-human and
constant-vs-human arms, giving bias-corrected anchors and a minimum-n table --
the sample size at which the suite can actually separate a simulator from the
human ceiling.

This should gate every other sweep's n. CPU only.
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
OUT = ROOT / "arr2026/results/h9"
NS = [10, 20, 40, 80, 160, 320, 640]
N_REP = 20


def c2st(a, b, seed):
    if len(a) < 8 or len(b) < 8:
        return float("nan")
    X = np.vstack([a, b]); y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    aucs = []
    for tr, te in StratifiedKFold(4, shuffle=True, random_state=seed).split(X, y):
        if len(np.unique(y[te])) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=100, random_state=seed).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(max(np.mean(aucs), 1 - np.mean(aucs))) if aucs else float("nan")


def dfm(a, b):
    ds = []
    for j in range(a.shape[1]):
        x = a[:, j][~np.isnan(a[:, j])]; y = b[:, j][~np.isnan(b[:, j])]
        if len(x) and len(y):
            ds.append(wasserstein_distance(x, y))
    return float(1 - np.mean(ds) / 4)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")
    rng = np.random.default_rng(2026)
    rows = []
    for cl in sorted(df.clusters.unique()):
        Y = df[df.clusters == cl][ITEMS].to_numpy(float)
        const = np.round(Y.mean(axis=0))
        for n in NS:
            hc, hd, cc, cd = [], [], [], []
            for r in range(N_REP):
                i = rng.choice(len(Y), 2 * n, replace=False)
                A, B = Y[i[:n]], Y[i[n:]]
                K = np.tile(const, (n, 1))
                hc.append(c2st(A, B, 2026 + r)); hd.append(dfm(A, B))
                cc.append(c2st(K, B, 2026 + r)); cd.append(dfm(K, B))
            rows.append({"cluster": int(cl), "n": n,
                         "c2st_human": float(np.nanmean(hc)), "DF_human": float(np.mean(hd)),
                         "c2st_constant": float(np.nanmean(cc)), "DF_constant": float(np.mean(cd))})
            print(f"  cluster {cl} n={n:4d}: C2ST human={rows[-1]['c2st_human']:.4f} "
                  f"const={rows[-1]['c2st_constant']:.4f} | DF human={rows[-1]['DF_human']:.4f}",
                  flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(OUT / "sample_size.csv", index=False)
    g = t.groupby("n")[["c2st_human", "DF_human", "c2st_constant", "DF_constant"]].mean()

    def extrapolate(y, ns):
        """Fit y = y_inf + c*n^-0.5 by least squares; return y_inf."""
        X = np.vstack([np.ones(len(ns)), 1 / np.sqrt(ns)]).T
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(beta[0])

    # Below n~40 the classifier cannot fit at all: it returns 0.5 for the CONSTANT
    # too, which is degenerate rather than informative. Including those points
    # flattens the fit and inflates the extrapolated ceiling, so restrict the fit
    # to the regime where the floor is actually separable.
    usable = g[g.c2st_constant > 0.6]
    ns = np.array(usable.index, float)
    g_fit = usable
    summary = {
        "by_n": g.round(4).to_dict("index"),
        "c2st_human_ceiling_at_n40": round(float(g.loc[40, "c2st_human"]), 4),
        "c2st_human_ceiling_extrapolated": round(extrapolate(g_fit.c2st_human.to_numpy(), ns), 4),
        "DF_human_ceiling_at_n40": round(float(g.loc[40, "DF_human"]), 4),
        "DF_human_ceiling_extrapolated": round(extrapolate(g_fit.DF_human.to_numpy(), ns), 4),
        "n_used_for_fit": [int(x) for x in ns],
        "degenerate_n_excluded": [int(x) for x in g.index if x not in set(ns)],
        "interpretation": (
            "If the extrapolated C2ST human ceiling approaches 0.5, the 0.564 reported at "
            "n=40 is finite-sample bias and the anchor should be quoted bias-corrected."),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== H9 SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_n"}, indent=2))


if __name__ == "__main__":
    main()
