"""H13: is the individuation models do produce pointed in the right direction?

VR_between measures how far apart a model spreads its persona means; it says
nothing about whether the spread matches WHICH respondent is which. A simulator
could reach VR_between = 0.25 with means uncorrelated to the truth, which is
dispersion without identity.

For each item we correlate, across the forty personas, the model's persona mean
against the oracle persona mean E[X_j | code] fitted on held-out respondents.
A correlation near 1 means the individuation is correct but too small, a scaling failure that
amplification could fix; r near 0 means it points nowhere, which amplification
would only make louder.
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


def channel_scores(cluster: int):
    base = ROOT / "src/prompt/mean_value_cluster"
    tr = json.loads((base / "traits.json").read_text(encoding="utf-8"))
    fa = json.loads((base / "facets.json").read_text(encoding="utf-8"))
    ck = str(cluster)
    return list(tr.get(ck, tr)) + list(fa.get(ck, fa))


def quantise(v):
    return np.clip([bisect.bisect_right(BOUNDS, x) - 1 for x in v], 0, N_LEVELS - 1)


def ridge(X, Y, Xp, lam=1e-3):
    Xc = np.hstack([X, np.ones((len(X), 1))])
    Xpc = np.hstack([Xp, np.ones((len(Xp), 1))])
    W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ Y)
    return Xpc @ W


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n-personas", type=int, default=40)
    a = ap.parse_args()
    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")

    for run in a.runs:
        rd = ROOT / a.results / run
        if not (rd / "summary.json").exists():
            print(f"{run:22s} MISSING"); continue
        rs, vrs = [], []
        for d in sorted(rd.glob("readout_cluster_*")):
            cl = int(d.name.rsplit("_", 1)[1])
            probs = np.load(d / "belief_probs.npy")
            sub_all = df[df.clusters == cl]
            personas = sub_all.iloc[: a.n_personas]
            fit_rows = sub_all.iloc[a.n_personas:]
            human_sd = np.nanstd(sub_all[ITEMS].to_numpy(float), axis=0)
            ok = human_sd > 0

            vals = np.arange(1, 6, dtype=float)
            mu = np.nansum(probs * vals[None, None, :], axis=2)     # (personas, items)

            chan = [c for c in channel_scores(cl) if c in df.columns]
            qf = np.column_stack([quantise(fit_rows[c].to_numpy(float)) for c in chan])
            qp = np.column_stack([quantise(personas[c].to_numpy(float)) for c in chan])
            oh_f = np.zeros((len(qf), len(chan) * N_LEVELS))
            oh_p = np.zeros((len(qp), len(chan) * N_LEVELS))
            for k in range(len(chan)):
                oh_f[np.arange(len(qf)), k * N_LEVELS + qf[:, k]] = 1.0
                oh_p[np.arange(len(qp)), k * N_LEVELS + qp[:, k]] = 1.0
            oracle = ridge(oh_f, fit_rows[ITEMS].to_numpy(float), oh_p)  # (personas, items)

            for j in np.where(ok)[0]:
                x, y = mu[:, j], oracle[:, j]
                if np.nanstd(x) < 1e-9 or np.nanstd(y) < 1e-9:
                    continue
                rs.append(float(np.corrcoef(x, y)[0, 1]))
            vrs.append(float(np.nanmean(np.sqrt(np.nanvar(mu, axis=0))[ok] / human_sd[ok])))

        r = float(np.nanmean(rs)); vr = float(np.nanmean(vrs))
        print(f"{run:22s} r={r:+.3f}  VR_betw={vr:.3f}  aligned={r*vr:+.3f} "
              f"(items={len(rs)})", flush=True)


if __name__ == "__main__":
    main()
