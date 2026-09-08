"""H12: where in the item space does individuation go missing?

Each cluster's prompt names five traits and five of the thirty facets. Twenty of
the 120 items therefore have their own facet stated explicitly; the other 100 are
covered only at trait level. If models individuate mainly where the prompt is
explicit, the shortfall is a failure to propagate coarse trait information; if
the shortfall is uniform, it is a failure to individuate at all.

Raw VR is confounded -- named-facet items are intrinsically more predictable from
the code -- so each subset is compared with its OWN ceiling, estimated on
held-out respondents exactly as in h11.
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


def norm(s: str) -> str:
    return s.lower().replace("facet_", "").replace("-", "_").replace(" ", "_").strip()


def channel_scores(cluster: int):
    base = ROOT / "src/prompt/mean_value_cluster"
    tr = json.loads((base / "traits.json").read_text(encoding="utf-8"))
    fa = json.loads((base / "facets.json").read_text(encoding="utf-8"))
    ck = str(cluster)
    return list(tr.get(ck, tr)), list(fa.get(ck, fa))


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
    key = pd.read_csv(ROOT / "data/IPIP-NEO/120/item_key.csv")
    item_facet = {int(r.item): norm(r.facet) for r in key.itertuples()}
    all_facets = [c for c in df.columns if c.startswith("facet_")]
    scores_all = TRAITS + all_facets

    for run in a.runs:
        rd = ROOT / a.results / run
        if not (rd / "summary.json").exists():
            print(f"{run:20s} MISSING"); continue
        agg = {k: [] for k in ("vr_named", "vr_other", "ceil_named", "ceil_other")}
        for d in sorted(rd.glob("readout_cluster_*")):
            cl = int(d.name.rsplit("_", 1)[1])
            probs = np.load(d / "belief_probs.npy")
            tr_k, fa_k = channel_scores(cl)
            named = {norm(f) for f in fa_k}
            mask_named = np.array([item_facet.get(i + 1, "") in named for i in range(120)])

            sub_all = df[df.clusters == cl]
            personas = sub_all.iloc[: a.n_personas]
            fit_rows = sub_all.iloc[a.n_personas:]
            human_sd = np.nanstd(sub_all[ITEMS].to_numpy(float), axis=0)
            ok = human_sd > 0

            vals = np.arange(1, 6, dtype=float)
            mu = np.nansum(probs * vals[None, None, :], axis=2)
            betw_sd = np.sqrt(np.nanvar(mu, axis=0))

            chan = [c for c in (tr_k + fa_k) if c in df.columns]
            qf = np.column_stack([quantise(fit_rows[c].to_numpy(float)) for c in chan])
            qp = np.column_stack([quantise(personas[c].to_numpy(float)) for c in chan])
            oh_f = np.zeros((len(qf), len(chan) * N_LEVELS))
            oh_p = np.zeros((len(qp), len(chan) * N_LEVELS))
            for k in range(len(chan)):
                oh_f[np.arange(len(qf)), k * N_LEVELS + qf[:, k]] = 1.0
                oh_p[np.arange(len(qp)), k * N_LEVELS + qp[:, k]] = 1.0
            pred = ridge(oh_f, fit_rows[ITEMS].to_numpy(float), oh_p)
            ceil_sd = np.nanstd(pred, axis=0)

            for label, m in (("named", mask_named & ok), ("other", (~mask_named) & ok)):
                agg[f"vr_{label}"].append(float(np.nanmean(betw_sd[m] / human_sd[m])))
                agg[f"ceil_{label}"].append(float(np.nanmean(ceil_sd[m] / human_sd[m])))

        r = {k: float(np.mean(v)) for k, v in agg.items()}
        en = r["vr_named"] / r["ceil_named"] if r["ceil_named"] else float("nan")
        eo = r["vr_other"] / r["ceil_other"] if r["ceil_other"] else float("nan")
        print(f"{run:20s} named: VR={r['vr_named']:.3f} ceil={r['ceil_named']:.3f} "
              f"eff={en:.2f} | other: VR={r['vr_other']:.3f} ceil={r['ceil_other']:.3f} "
              f"eff={eo:.2f}", flush=True)


if __name__ == "__main__":
    main()
