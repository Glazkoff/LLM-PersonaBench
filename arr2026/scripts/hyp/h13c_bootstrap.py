"""H13c: joint persona bootstrap for the aligned amplitude A.

A is a point estimate over 40 fixed personas per cluster, so comparisons between
models and the model-vs-human reference carry no stated uncertainty. This
resamples persona ROWS jointly -- the same resampled indices are applied to the
model's persona means, the decoder's predictions and the actual respondents, so
paired comparisons stay paired -- and recomputes A in every draw.

The decoder fit is held fixed across draws: it is estimated on respondents
excluding these 40, so the intervals condition on that fit and describe
persona-sampling uncertainty only. That is the quantity the comparative claims
need; a fit-refitting interval would be wider and is not what is reported.
"""
import argparse
import bisect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _orientation import to_human_orientation  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ITEMS = [f"i{i}" for i in range(1, 121)]
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
B, L = [0, 20, 40, 60, 80, 100], 5


def channel(cl):
    b = ROOT / "src/prompt/mean_value_cluster"
    tr = json.loads((b / "traits.json").read_text())
    fa = json.loads((b / "facets.json").read_text())
    return list(tr.get(str(cl), tr)) + list(fa.get(str(cl), fa))


def q(v):
    return np.clip([bisect.bisect_right(B, x) - 1 for x in v], 0, L - 1)


def oh(a, k):
    m = np.zeros((len(a), k * L))
    for j in range(k):
        m[np.arange(len(a)), j * L + a[:, j]] = 1.0
    return m


def ridge(X, Y, Xp, lam=1e-3):
    Xc = np.hstack([X, np.ones((len(X), 1))])
    Xpc = np.hstack([Xp, np.ones((len(Xp), 1))])
    return Xpc @ np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ Y)


def A_of(M, oracle, hsd, ok, rows):
    """Aligned amplitude on a given set of persona rows."""
    Mr, Or = M[rows], oracle[rows]
    msd = np.sqrt(np.nanvar(Mr, axis=0))
    out = []
    for j in np.where(ok)[0]:
        if np.nanstd(Mr[:, j]) < 1e-9 or np.nanstd(Or[:, j]) < 1e-9:
            continue
        out.append(float(np.corrcoef(Mr[:, j], Or[:, j])[0, 1]) * float(msd[j] / hsd[j]))
    return float(np.nanmean(out)) if out else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n-personas", type=int, default=40)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=260909)
    ap.add_argument("--pairs", nargs="*", default=[],
                    help="run:run comparisons to report as PAIRED differences")
    a = ap.parse_args()

    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")
    ctx = {}
    for cl in (0, 1, 2, 3):
        sub = df[df.clusters == cl]
        per, fit = sub.iloc[: a.n_personas], sub.iloc[a.n_personas:]
        hsd = np.nanstd(sub[ITEMS].to_numpy(float), axis=0)
        c = [x for x in channel(cl) if x in df.columns]
        qf = np.column_stack([q(fit[x].to_numpy(float)) for x in c])
        qp = np.column_stack([q(per[x].to_numpy(float)) for x in c])
        ctx[cl] = {"oracle": ridge(oh(qf, len(c)), fit[ITEMS].to_numpy(float), oh(qp, len(c))),
                   "hsd": hsd, "ok": hsd > 0,
                   "actual": per[ITEMS].to_numpy(float)}
        print(f"[decoder fixed for cluster {cl}]", flush=True)

    rng = np.random.default_rng(a.seed)
    idx = {cl: [rng.integers(0, a.n_personas, a.n_personas) for _ in range(a.draws)]
           for cl in ctx}          # shared draws: paired across every quantity

    # human reference under the same resampling
    href = []
    for b in range(a.draws):
        href.append(np.mean([A_of(ctx[cl]["actual"], ctx[cl]["oracle"], ctx[cl]["hsd"],
                                  ctx[cl]["ok"], idx[cl][b]) for cl in ctx]))
    href = np.array(href)
    lo, hi = np.percentile(href, [2.5, 97.5])
    _hr_ci = (lo, hi)   # keep the interval; the point estimate is printed below

    href_point = np.mean([A_of(ctx[cl]["actual"], ctx[cl]["oracle"], ctx[cl]["hsd"],
                               ctx[cl]["ok"], np.arange(a.n_personas)) for cl in ctx])
    print(f"human reference A (original sample) = {href_point:.4f}\n")
    store, points = {}, {}
    print(f"{'run':22s} {'A':>8s} {'95% CI':>18s} {'A/human':>9s} {'CI(ratio)':>18s}")
    for run in a.runs:
        rd = None
        for base in a.results:
            if (ROOT / base / run / "summary.json").exists():
                rd = ROOT / base / run
                break
        if rd is None:
            print(f"{run:22s} MISSING"); continue
        per_cl = {}
        for d in sorted(rd.glob("readout_cluster_*")):
            cl = int(d.name.rsplit("_", 1)[1])
            probs = np.load(d / "belief_probs.npy")
            mu = np.nansum(probs * np.arange(1, 6, dtype=float)[None, None, :], axis=2)
            per_cl[cl] = to_human_orientation(mu)
        if not per_cl:
            print(f"{run:22s} NO TENSORS"); continue
        draws = np.array([
            np.mean([A_of(per_cl[cl], ctx[cl]["oracle"], ctx[cl]["hsd"], ctx[cl]["ok"], idx[cl][b])
                     for cl in per_cl])
            for b in range(a.draws)])
        ratio = draws / href[: len(draws)]
        l1, h1 = np.percentile(draws, [2.5, 97.5])
        l2, h2 = np.percentile(ratio, [2.5, 97.5])
        # Report the original-sample estimate alongside the percentile interval;
        # the bootstrap mean is a different estimator and mixing the two silently
        # was what produced .183 next to .180.
        point = np.mean([A_of(per_cl[cl], ctx[cl]["oracle"], ctx[cl]["hsd"],
                              ctx[cl]["ok"], np.arange(a.n_personas)) for cl in per_cl])
        store[run] = draws
        points[run] = point
        print(f"{run:22s} {point:8.4f} [{l1:7.4f},{h1:7.4f}] "
              f"{point/href_point:9.3f} [{l2:7.3f},{h2:7.3f}]", flush=True)

    # Paired differences use the SAME draws, so they are not the marginal
    # intervals differenced: two overlapping marginals can still have a
    # difference interval that excludes zero.
    for spec in a.pairs:
        x, y = spec.split(":")
        if x in store and y in store:
            d = store[x] - store[y]
            lo, hi = np.percentile(d, [2.5, 97.5])
            sig = "resolved" if (lo > 0 or hi < 0) else "not resolved"
            # original-sample difference with the bootstrap interval, so the point
            # estimate and the interval are not two different estimators
            point_d = points[x] - points[y]
            print(f"  paired {x} - {y}: {point_d:+.4f} [{lo:+.4f},{hi:+.4f}]  {sig}",
                  flush=True)


if __name__ == "__main__":
    main()
