"""H13b: is the alignment correlation r robust to how the oracle is specified?

h13 measured r against one oracle: additive ridge on the one-hot prompt code. If
that oracle is missing structure the models are actually tracking, r would be
understated. This recomputes r under five targets:

  onehot   additive ridge on the 10-score, 5-level code   (the h13 oracle)
  boost    boosted trees on the same code                 (adds interactions)
  raw      ridge on the 10 continuous scores              (no quantisation)
  all35    additive ridge on all 35 scores, one-hot       (richer than the prompt)
  actual   the personas' own human answers                (no oracle at all)

`actual` is the assumption-free check: it needs no model of the code, only the
respondents the personas actually are. It is attenuated by individual response
noise, so we also report how well each ORACLE tracks `actual`, which sets the
scale a simulator could plausibly reach.

Oracles do not depend on the model, so each is fitted once per cluster and
correlated against every model.
"""
import argparse
import bisect
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ITEMS = [f"i{i}" for i in range(1, 121)]
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
BOUNDS = [0, 20, 40, 60, 80, 100]
N_LEVELS = 5


def reverse_keyed_mask():
    """Items stored reverse-recoded in the corpus.

    The corpus recodes 55 of the 120 items; models answer the literal item. VR
    is invariant to a per-item sign flip, so the belief-variance results are
    unaffected, but a CORRELATION is not: leaving this out flips the sign on 55
    of 120 items and drags any alignment estimate toward zero. h3 already maps
    models into the human orientation; these analyses must do the same.
    """
    import csv
    key = list(csv.DictReader(open(ROOT / "data/IPIP-NEO/120/item_key.csv")))
    neg = np.zeros(120, dtype=bool)
    for r in key:
        if str(r["reverse"]).strip().lower() == "true":
            neg[int(r["item"]) - 1] = True
    return neg


def to_human_orientation(mu, neg):
    """Map model item means onto the corpus's recoded scale."""
    out = mu.copy()
    out[:, neg] = 6.0 - out[:, neg]
    return out


def channel_scores(cluster: int):
    base = ROOT / "src/prompt/mean_value_cluster"
    tr = json.loads((base / "traits.json").read_text(encoding="utf-8"))
    fa = json.loads((base / "facets.json").read_text(encoding="utf-8"))
    ck = str(cluster)
    return list(tr.get(ck, tr)) + list(fa.get(ck, fa))


def quantise(v):
    return np.clip([bisect.bisect_right(BOUNDS, x) - 1 for x in v], 0, N_LEVELS - 1)


def onehot(q, k):
    m = np.zeros((len(q), k * N_LEVELS))
    for j in range(k):
        m[np.arange(len(q)), j * N_LEVELS + q[:, j]] = 1.0
    return m


def ridge(X, Y, Xp, lam=1e-3):
    Xc = np.hstack([X, np.ones((len(X), 1))])
    Xpc = np.hstack([Xp, np.ones((len(Xp), 1))])
    W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ Y)
    return Xpc @ W


def colwise_r(A, B, ok):
    """Mean over items of the across-persona correlation between two (n,120) sets."""
    out = []
    for j in np.where(ok)[0]:
        x, y = A[:, j], B[:, j]
        if np.nanstd(x) < 1e-9 or np.nanstd(y) < 1e-9:
            continue
        out.append(float(np.corrcoef(x, y)[0, 1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True,
                    help="one or more results dirs to search for runs")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n-personas", type=int, default=40)
    ap.add_argument("--boost", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")
    NEG = reverse_keyed_mask()
    all_scores = TRAITS + [c for c in df.columns if c.startswith("facet_")]

    oracles, oks, actuals = {}, {}, {}
    for cl in (0, 1, 2, 3):
        sub_all = df[df.clusters == cl]
        personas = sub_all.iloc[: a.n_personas]
        fit = sub_all.iloc[a.n_personas:]
        Yf = fit[ITEMS].to_numpy(float)
        human_sd = np.nanstd(sub_all[ITEMS].to_numpy(float), axis=0)
        oks[cl] = human_sd > 0
        actuals[cl] = personas[ITEMS].to_numpy(float)

        chan = [c for c in channel_scores(cl) if c in df.columns]
        qf = np.column_stack([quantise(fit[c].to_numpy(float)) for c in chan])
        qp = np.column_stack([quantise(personas[c].to_numpy(float)) for c in chan])
        d = {}
        d["onehot"] = ridge(onehot(qf, len(chan)), Yf, onehot(qp, len(chan)))
        d["raw"] = ridge(fit[chan].to_numpy(float), Yf, personas[chan].to_numpy(float))
        q35f = np.column_stack([quantise(fit[c].to_numpy(float)) for c in all_scores])
        q35p = np.column_stack([quantise(personas[c].to_numpy(float)) for c in all_scores])
        d["all35"] = ridge(onehot(q35f, len(all_scores)), Yf, onehot(q35p, len(all_scores)))
        if a.boost:
            from sklearn.ensemble import HistGradientBoostingRegressor
            pred = np.zeros((len(qp), len(ITEMS)))
            for j in range(len(ITEMS)):
                if not oks[cl][j]:
                    continue
                m = HistGradientBoostingRegressor(
                    max_iter=80, max_depth=6, learning_rate=0.1,
                    categorical_features=list(range(len(chan))), random_state=0)
                m.fit(qf, Yf[:, j])
                pred[:, j] = m.predict(qp)
            d["boost"] = pred
        oracles[cl] = d
        print(f"[oracles built for cluster {cl}]", flush=True)

    # how well does each oracle itself track the actual respondents?
    ref = {k: [] for k in oracles[0]}
    for cl in oracles:
        for k, pred in oracles[cl].items():
            ref[k] += colwise_r(pred, actuals[cl], oks[cl])
    print("\noracle vs actual respondents (the scale a simulator could reach):")
    for k, v in ref.items():
        print(f"  {k:8s} r={np.mean(v):+.3f}")

    print("\nmodel alignment under each oracle:")
    names = list(oracles[0]) + ["actual"]
    print(f"{'run':22s} " + " ".join(f"{n:>8s}" for n in names))
    for run in a.runs:
        rd = None
        for base in a.results:
            if (ROOT / base / run / "summary.json").exists():
                rd = ROOT / base / run
                break
        if rd is None:
            print(f"{run:22s} MISSING"); continue
        acc = {n: [] for n in names}
        for d in sorted(rd.glob("readout_cluster_*")):
            cl = int(d.name.rsplit("_", 1)[1])
            probs = np.load(d / "belief_probs.npy")
            mu = np.nansum(probs * np.arange(1, 6, dtype=float)[None, None, :], axis=2)
            mu = to_human_orientation(mu, NEG)
            for k, pred in oracles[cl].items():
                acc[k] += colwise_r(mu, pred, oks[cl])
            acc["actual"] += colwise_r(mu, actuals[cl], oks[cl])
        print(f"{run:22s} " + " ".join(f"{np.mean(acc[n]):+8.3f}" for n in names), flush=True)


if __name__ == "__main__":
    main()
