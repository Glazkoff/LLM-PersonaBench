"""H11: the information ceiling of the persona prompt.

The prompt individuates a respondent only through a 5-level quantisation of each
of 35 personality scores (5 traits + 30 facets): `get_modifier_bisect` bins the
0--100 score on boundaries [0,20,40,60,80,100]. Every other sentence in the
system prompt is shared by all personas in a cluster.

That imposes a ceiling on individuation which is a property of the *conditioning
scheme*, not of any model: two respondents landing in the same 35 bins receive
byte-identical prompts and must receive identical beliefs. This estimates that
ceiling from the human data alone, on exactly the personas the belief runs used,
so the number is directly comparable to the measured VR_between.

Reported per cluster and averaged:
  ceiling_quant : SD of E[item | quantised 35-bin code] / human SD   <- the real channel
  ceiling_raw   : SD of E[item | 35 continuous scores] / human SD    <- if the prompt sent exact numbers
  cells         : distinct 35-bin codes among the 40 personas        <- is the channel degenerate?
Both ceilings are estimated with an additive model and, optionally, with boosted
trees, which capture interactions and so give a tighter (higher) estimate.
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


def channel_scores(cluster: int):
    """The scores the prompt actually mentions for this cluster.

    Crucially this is not all 35: each cluster's genotype names 5 traits and only
    5 of the 30 facets, so the individuating channel is 10 numbers, each written
    as one of 5 words.
    """
    base = ROOT / "src/prompt/mean_value_cluster"
    tr = json.loads((base / "traits.json").read_text(encoding="utf-8"))
    fa = json.loads((base / "facets.json").read_text(encoding="utf-8"))
    ck = str(cluster)
    return list(tr.get(ck, tr)) + list(fa.get(ck, fa))
BOUNDS = [0, 20, 40, 60, 80, 100]
N_LEVELS = 5


def quantise(v: np.ndarray) -> np.ndarray:
    """Replicate get_modifier_bisect: bin a 0-100 score into 5 levels."""
    idx = np.array([bisect.bisect_right(BOUNDS, x) - 1 for x in v])
    return np.clip(idx, 0, N_LEVELS - 1)


def ridge_fit_predict(X: np.ndarray, Y: np.ndarray, Xp: np.ndarray, lam: float = 1e-3):
    """Closed-form ridge; Y is (n, n_items). Returns predictions for Xp."""
    Xc = np.hstack([X, np.ones((len(X), 1))])
    Xpc = np.hstack([Xp, np.ones((len(Xp), 1))])
    A = Xc.T @ Xc + lam * np.eye(Xc.shape[1])
    B = Xc.T @ Y
    W = np.linalg.solve(A, B)
    return Xpc @ W


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-personas", type=int, default=40)
    ap.add_argument("--clusters", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--boost", action="store_true", help="also fit boosted trees")
    ap.add_argument("--out", default="arr2026/results/h11_prompt_ceiling")
    a = ap.parse_args()

    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")
    facets = [c for c in df.columns if c.startswith("facet_")]
    scores = TRAITS + facets
    print(f"{len(scores)} conditioning scores ({len(TRAITS)} traits + {len(facets)} facets)",
          flush=True)
    print("score range:", float(df[scores].min().min()), "-", float(df[scores].max().max()),
          flush=True)

    rows = []
    for cl in a.clusters:
        sub_all = df[df.clusters == cl]
        personas = sub_all.iloc[: a.n_personas]
        # Hold the personas out of every fit: they are the evaluation set, and a
        # ceiling that a model is being measured against must not be fitted on them.
        fit_rows = sub_all.iloc[a.n_personas:]
        Y = fit_rows[ITEMS].to_numpy(float)
        Y_eval_sd = np.nanstd(sub_all[ITEMS].to_numpy(float), axis=0)
        human_sd = Y_eval_sd          # human SD over the whole cluster, as the runs used
        ok = human_sd > 0

        chan = [c for c in channel_scores(cl) if c in df.columns]
        raw_all = fit_rows[chan].to_numpy(float)
        raw_p = personas[chan].to_numpy(float)
        q_all = np.column_stack([quantise(raw_all[:, k]) for k in range(raw_all.shape[1])])
        q_p = np.column_stack([quantise(raw_p[:, k]) for k in range(raw_p.shape[1])])

        cells = len({tuple(r) for r in q_p})

        # one-hot of the quantised code = the literal prompt channel
        oh_all = np.zeros((len(q_all), len(chan) * N_LEVELS))
        oh_p = np.zeros((len(q_p), len(chan) * N_LEVELS))
        for k in range(len(chan)):
            oh_all[np.arange(len(q_all)), k * N_LEVELS + q_all[:, k]] = 1.0
            oh_p[np.arange(len(q_p)), k * N_LEVELS + q_p[:, k]] = 1.0

        pred_q = ridge_fit_predict(oh_all, Y, oh_p)
        pred_r = ridge_fit_predict(raw_all, Y, raw_p)
        vr_q = float(np.nanmean(np.nanstd(pred_q, axis=0)[ok] / human_sd[ok]))
        vr_r = float(np.nanmean(np.nanstd(pred_r, axis=0)[ok] / human_sd[ok]))

        vr_qb = float("nan")
        if a.boost:
            from sklearn.ensemble import HistGradientBoostingRegressor
            preds = np.zeros((len(q_p), len(ITEMS)))
            for j in range(len(ITEMS)):
                if not ok[j]:
                    continue
                m = HistGradientBoostingRegressor(
                    max_iter=80, max_depth=6, learning_rate=0.1,
                    categorical_features=list(range(len(chan))), random_state=0)
                m.fit(q_all, Y[:, j])
                preds[:, j] = m.predict(q_p)
            vr_qb = float(np.nanmean(np.nanstd(preds, axis=0)[ok] / human_sd[ok]))

        raw35_all = fit_rows[scores].to_numpy(float)
        raw35_p = personas[scores].to_numpy(float)
        q35_all = np.column_stack([quantise(raw35_all[:, k]) for k in range(raw35_all.shape[1])])
        q35_p = np.column_stack([quantise(raw35_p[:, k]) for k in range(raw35_p.shape[1])])
        oh35_all = np.zeros((len(q35_all), len(scores) * N_LEVELS))
        oh35_p = np.zeros((len(q35_p), len(scores) * N_LEVELS))
        for k in range(len(scores)):
            oh35_all[np.arange(len(q35_all)), k * N_LEVELS + q35_all[:, k]] = 1.0
            oh35_p[np.arange(len(q35_p)), k * N_LEVELS + q35_p[:, k]] = 1.0
        pred35 = ridge_fit_predict(oh35_all, Y, oh35_p)
        vr_35 = float(np.nanmean(np.nanstd(pred35, axis=0)[ok] / human_sd[ok]))
        pred35r = ridge_fit_predict(raw35_all, Y, raw35_p)
        vr_35r = float(np.nanmean(np.nanstd(pred35r, axis=0)[ok] / human_sd[ok]))

        # how many of the personas are literally indistinguishable to the model?
        codes = [tuple(r) for r in q_p]
        collided = len(codes) - len(set(codes))

        rows.append({"cluster": cl, "n_fit": int(len(fit_rows)),
                     "channel_scores": len(chan),
                     "identical_prompt_personas": collided,
                     "ceiling_all35_bins": vr_35, "ceiling_all35_raw": vr_35r,
                     "distinct_cells_among_personas": cells,
                     "ceiling_quant_additive": vr_q,
                     "ceiling_quant_boosted": vr_qb,
                     "ceiling_raw_additive": vr_r})
        print(f"cluster {cl}: n={len(sub_all)} chan={len(chan)} cells={cells}/{a.n_personas} "
              f"collisions={collided} | ceiling_quant={vr_q:.3f} boosted={vr_qb:.3f} "
              f"ceiling_raw={vr_r:.3f} | all35_bins={vr_35:.3f} all35_raw={vr_35r:.3f}",
              flush=True)

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(rows)
    t.to_csv(out / "ceilings.csv", index=False)
    summary = {
        "ceiling_quant_additive_mean": round(float(t.ceiling_quant_additive.mean()), 4),
        "ceiling_quant_boosted_mean": round(float(t.ceiling_quant_boosted.mean()), 4),
        "ceiling_raw_additive_mean": round(float(t.ceiling_raw_additive.mean()), 4),
        "ceiling_all35_bins_mean": round(float(t.ceiling_all35_bins.mean()), 4),
        "ceiling_all35_raw_mean": round(float(t.ceiling_all35_raw.mean()), 4),
        "channel_scores": int(t.channel_scores.iloc[0]),
        "identical_prompt_personas_total": int(t.identical_prompt_personas.sum()),
        "distinct_cells_mean": float(t.distinct_cells_among_personas.mean()),
        "n_personas": a.n_personas,
        "observed_max_VR_between": 0.295,
        "observed_min_VR_between": 0.126,
        "note": ("Ceilings are estimated on the same personas the belief runs used, so "
                 "they are directly comparable to VR_between. The additive figure is a "
                 "lower bound on the channel; the boosted figure adds interactions."),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
