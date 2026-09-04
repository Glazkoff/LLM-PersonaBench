"""
H2 -- Is C2ST reading trait content, or just response style?

Our replacement metric reports C2ST-AUC 0.983 for LLM personas against a human
ceiling of 0.564. The obvious reviewer attack is that the classifier is not
detecting anything about personality at all -- it is detecting formatting or
response style (LLMs never straightline, never make reverse-keying errors, avoid
or over-use the scale endpoints). If a handful of content-free scalars reach the
same AUC, our own metric is measuring the least interesting difference.

Feature ladder, all scored with the same classifier and the same human ceiling
and constant floor:
  A  full 120 items                       (reproduce 0.983)
  B  5 style scalars only                 (mean, sd, extreme rate, longest run,
                                           forward-vs-reverse consistency)
  C  120 items, per-respondent ipsatised  (each respondent's mean and sd removed,
                                           so only within-person pattern remains)
  D  30 facet scores only                 (trait content, style largely removed)
  E  forward-keyed items only / reverse-keyed items only

If B is near chance and C stays high, C2ST is reading trait structure and the
metric survives. If B ~= A and C collapses, it is a style detector and we must
say so ourselves.

CPU only. No LLM calls.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ITEMS = [f"i{i}" for i in range(1, 121)]
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "arr2026/results/h2"
SEED, N_REF, N_REP = 2026, 40, 10


def load_keys():
    """Return (reverse_mask over 120 items, facet_id per item) from the IPIP key."""
    try:
        k = pd.read_excel(ROOT / "data/PAlign/IPIP-NEO-ItemKey.xls")
        k = k[k["Short#"].notna()].copy()
        k["Short#"] = k["Short#"].astype(int)
        k = k[k["Short#"] <= 120].sort_values("Short#")
        rev = np.zeros(120, dtype=bool)
        facet = np.full(120, "", dtype=object)
        for _, r in k.iterrows():
            j = int(r["Short#"]) - 1
            rev[j] = str(r["Sign"]).strip().startswith("-")
            facet[j] = str(r["Key"]).strip()
        return rev, facet
    except Exception as e:
        print("WARN: item key unavailable:", repr(e)[:120])
        return np.zeros(120, dtype=bool), np.full(120, "", dtype=object)


REVERSE, FACET = load_keys()


def c2st(a, b, seed=SEED):
    if len(a) < 8 or len(b) < 8 or a.shape[1] == 0:
        return float("nan")
    X = np.vstack([a, b]); y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    aucs = []
    for tr, te in StratifiedKFold(4, shuffle=True, random_state=seed).split(X, y):
        if len(np.unique(y[te])) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=120, random_state=seed).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(max(np.mean(aucs), 1 - np.mean(aucs))) if aucs else float("nan")


def style_features(M):
    """5 content-free response-style scalars per respondent."""
    out = []
    for r in M:
        v = r[~np.isnan(r)]
        if len(v) == 0:
            out.append([np.nan] * 5); continue
        # longest run of identical consecutive answers
        run = best = 1
        for i in range(1, len(v)):
            run = run + 1 if v[i] == v[i - 1] else 1
            best = max(best, run)
        fwd = r[~REVERSE]; rev = r[REVERSE]
        fwd = fwd[~np.isnan(fwd)]; rev = rev[~np.isnan(rev)]
        consistency = (np.nanmean(fwd) + np.nanmean(rev) - 6.0) if len(fwd) and len(rev) else np.nan
        out.append([v.mean(), v.std(), np.mean((v == 1) | (v == 5)), best, consistency])
    return np.array(out, float)


def ipsatise(M):
    mu = np.nanmean(M, axis=1, keepdims=True)
    sd = np.nanstd(M, axis=1, keepdims=True)
    return (M - mu) / np.where(sd < 1e-9, 1.0, sd)


def facet_scores(M):
    """30 facet means per respondent (4 items each), reverse-scored."""
    if (FACET == "").all():
        return np.zeros((len(M), 0))
    Mr = M.copy()
    Mr[:, REVERSE] = 6.0 - Mr[:, REVERSE]
    cols = []
    for f in sorted(set(FACET[FACET != ""])):
        idx = np.where(FACET == f)[0]
        cols.append(np.nanmean(Mr[:, idx], axis=1))
    return np.vstack(cols).T


VARIANTS = {
    "A_full_items":    lambda M: M,
    "B_style_only":    style_features,
    "C_ipsatised":     ipsatise,
    "D_facets_only":   facet_scores,
    "E_forward_only":  lambda M: M[:, ~REVERSE],
    "E_reverse_only":  lambda M: M[:, REVERSE],
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")
    rows = []
    for cdir in sorted((ROOT / "results_experiments/evoprompt_iter2").glob("*/*/cluster_*")):
        cid = int(cdir.name.split("_")[1]); model = cdir.parent.parent.name
        f = cdir / "after_optimization_test_answers.csv"
        tr_f, te_f = cdir / "train_case_ids.csv", cdir / "test_case_ids.csv"
        if not (f.exists() and te_f.exists() and tr_f.exists()):
            continue
        M = pd.read_csv(f)[ITEMS].to_numpy(float)
        if np.isnan(M).mean() > 0.5 or len(np.unique(M[~np.isnan(M)])) < 2:
            continue
        te = set(pd.read_csv(te_f)["case"]); tr = set(pd.read_csv(tr_f)["case"])
        cl = df[df.clusters == cid]
        pool = cl[~cl.case.isin(te | tr)][ITEMS].to_numpy(float)
        tri = cl[cl.case.isin(tr)][ITEMS].to_numpy(float)
        if len(pool) < 2 * N_REF or len(tri) == 0:
            continue
        const = np.tile(np.round(tri.mean(axis=0)), (N_REF, 1))

        for name, fn in VARIANTS.items():
            am, ah, ac = [], [], []
            for r in range(N_REP):
                rr = np.random.default_rng(SEED + r)
                idx = rr.choice(len(pool), 2 * N_REF, replace=False)
                ra, rb = pool[idx[:N_REF]], pool[idx[N_REF:]]
                am.append(c2st(fn(M), fn(ra), SEED + r))
                ah.append(c2st(fn(rb), fn(ra), SEED + r))
                ac.append(c2st(fn(const), fn(ra), SEED + r))
            rows.append({"model": model, "cluster": cid, "variant": name,
                         "c2st_model": float(np.nanmean(am)),
                         "c2st_human_ceiling": float(np.nanmean(ah)),
                         "c2st_constant_floor": float(np.nanmean(ac))})
            print(f"  {model:13s} c{cid} {name:16s} "
                  f"model={rows[-1]['c2st_model']:.3f} "
                  f"human={rows[-1]['c2st_human_ceiling']:.3f} "
                  f"const={rows[-1]['c2st_constant_floor']:.3f}", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(OUT / "c2st_ablation.csv", index=False)
    agg = t.groupby("variant")[["c2st_model", "c2st_human_ceiling", "c2st_constant_floor"]].mean().round(4)
    full = float(agg.loc["A_full_items", "c2st_model"])
    style = float(agg.loc["B_style_only", "c2st_model"])
    summary = {
        "by_variant": agg.to_dict("index"),
        "style_recovers_fraction_of_full_AUC_above_chance":
            round((style - 0.5) / max(full - 0.5, 1e-9), 3),
        "verdict": ("STYLE-DRIVEN: a handful of content-free scalars reproduce most of "
                    "the separability; C2ST must be reported with this caveat."
                    if (style - 0.5) / max(full - 0.5, 1e-9) > 0.8 else
                    "TRAIT-BEARING: style scalars do not reproduce the separability, so "
                    "C2ST is reading more than formatting."),
        "n_cells": int(t.cluster.count() / max(len(VARIANTS), 1)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== H2 SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
