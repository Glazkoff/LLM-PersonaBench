"""
E1 -- Cluster validity for the IPIP-NEO-120 facet clustering.

Answers the objection raised by all three NeurIPS reviewers and the AC:
k=4 was never justified, cluster stability was never measured, and the
clusters were never checked against demographics.

Outputs (arr2026/results/e1/):
  k_selection.csv        inertia / silhouette / CH / DB / gap over k=2..15
  stability.csv          bootstrap ARI per k, per preprocessing
  preprocessing.csv      how k-selection moves under raw / z-score / ipsative
  algorithms.csv         kmeans vs GMM vs Ward agreement with the shipped labels
  demographics.csv       Cramer's V of cluster vs sex / age band / country
  summary.json           machine-readable roll-up
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

FACETS = [
    "facet_imagination", "facet_artistic_interests", "facet_emotionality",
    "facet_adventurousness", "facet_intellect", "facet_liberalism",
    "facet_self_efficacy", "facet_orderliness", "facet_dutifulness",
    "facet_achievement_striving", "facet_self_discipline", "facet_cautiousness",
    "facet_friendliness", "facet_gregariousness", "facet_assertiveness",
    "facet_activity_level", "facet_excitement_seeking", "facet_cheerfulness",
    "facet_trust", "facet_morality", "facet_altruism", "facet_cooperation",
    "facet_modesty", "facet_sympathy",
    "facet_anxiety", "facet_anger", "facet_depression",
    "facet_self_consciousness", "facet_immoderation", "facet_vulnerability",
]

K_RANGE = list(range(2, 16))


def preprocess(X, mode):
    if mode == "raw":
        return X.copy()
    if mode == "zscore":
        return StandardScaler().fit_transform(X)
    if mode == "ipsative":
        # within-person centering: removes individual response style, the
        # standard psychometric control the paper never applied
        return X - X.mean(axis=1, keepdims=True)
    raise ValueError(mode)


def fit_kmeans(X, k, seed):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)


def gap_statistic(X, k, seed, n_ref=5):
    """Tibshirani gap statistic against uniform references on the data bbox."""
    rng = np.random.default_rng(seed)
    log_wk = np.log(fit_kmeans(X, k, seed).inertia_ + 1e-12)
    mins, maxs = X.min(axis=0), X.max(axis=0)
    refs = []
    for i in range(n_ref):
        Xr = rng.uniform(mins, maxs, size=X.shape)
        refs.append(np.log(fit_kmeans(Xr, k, seed + i).inertia_ + 1e-12))
    return float(np.mean(refs) - log_wk), float(np.std(refs))


def cramers_v(confusion):
    chi2 = 0.0
    n = confusion.values.sum()
    exp = np.outer(confusion.sum(axis=1), confusion.sum(axis=0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((confusion.values - exp) ** 2 / exp)
    r, c = confusion.shape
    denom = n * (min(r - 1, c - 1) or 1)
    return float(np.sqrt(chi2 / denom))


def permutation_p(labels, factor, observed, n_perm, seed):
    rng = np.random.default_rng(seed)
    lab = np.asarray(labels)
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(lab)
        v = cramers_v(pd.crosstab(pd.Series(perm), factor))
        if v >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/df_ipipneo_120_clusters")
    ap.add_argument("--out", default="arr2026/results/e1")
    ap.add_argument("--sil-sample", type=int, default=20000,
                    help="subsample for silhouette (O(n^2) memory)")
    ap.add_argument("--fit-sample", type=int, default=0,
                    help="0 = fit on all rows")
    ap.add_argument("--boot", type=int, default=50, help="bootstrap resamples for stability")
    ap.add_argument("--boot-sample", type=int, default=50000)
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = pd.read_csv(args.data)
    print(f"loaded {len(df)} rows", flush=True)
    shipped = df["clusters"].to_numpy()
    X_raw = df[FACETS].to_numpy(dtype=np.float64)

    if args.fit_sample and args.fit_sample < len(df):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(df), args.fit_sample, replace=False)
        X_raw, shipped = X_raw[idx], shipped[idx]
        print(f"fit subsample -> {len(X_raw)}", flush=True)

    rng = np.random.default_rng(args.seed)
    sil_idx = rng.choice(len(X_raw), min(args.sil_sample, len(X_raw)), replace=False)

    # ---------- k selection, across preprocessing ----------
    rows = []
    for mode in ("raw", "zscore", "ipsative"):
        X = preprocess(X_raw, mode)
        for k in K_RANGE:
            km = fit_kmeans(X, k, args.seed)
            lab = km.labels_
            gap, gap_sd = gap_statistic(X[sil_idx], k, args.seed)
            rows.append({
                "preprocessing": mode,
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(X[sil_idx], lab[sil_idx])),
                "calinski_harabasz": float(calinski_harabasz_score(X, lab)),
                "davies_bouldin": float(davies_bouldin_score(X, lab)),
                "gap": gap,
                "gap_sd": gap_sd,
                "ari_vs_shipped": float(adjusted_rand_score(shipped, lab)),
            })
            print(f"  [{mode}] k={k} sil={rows[-1]['silhouette']:.4f} "
                  f"db={rows[-1]['davies_bouldin']:.3f} gap={gap:.4f} "
                  f"ari_vs_shipped={rows[-1]['ari_vs_shipped']:.3f}", flush=True)
    ksel = pd.DataFrame(rows)
    ksel.to_csv(out / "k_selection.csv", index=False)

    # ---------- bootstrap stability ----------
    srows = []
    for mode in ("raw", "zscore"):
        X = preprocess(X_raw, mode)
        for k in K_RANGE:
            aris = []
            for b in range(args.boot):
                r = np.random.default_rng(args.seed + b)
                i1 = r.choice(len(X), args.boot_sample, replace=True)
                i2 = r.choice(len(X), args.boot_sample, replace=True)
                m1 = fit_kmeans(X[i1], k, args.seed + b)
                m2 = fit_kmeans(X[i2], k, args.seed + 1000 + b)
                # compare on the full set so labellings are comparable
                aris.append(adjusted_rand_score(m1.predict(X[sil_idx]),
                                                m2.predict(X[sil_idx])))
            srows.append({
                "preprocessing": mode, "k": k,
                "ari_mean": float(np.mean(aris)),
                "ari_std": float(np.std(aris)),
                "ari_p05": float(np.percentile(aris, 5)),
                "ari_p95": float(np.percentile(aris, 95)),
                "n_boot": args.boot,
            })
            print(f"  stability [{mode}] k={k} ARI={srows[-1]['ari_mean']:.3f}"
                  f" +-{srows[-1]['ari_std']:.3f}", flush=True)
    stab = pd.DataFrame(srows)
    stab.to_csv(out / "stability.csv", index=False)

    # ---------- alternative algorithms at k=4 ----------
    arows = []
    Xz = preprocess(X_raw, "zscore")
    sub = rng.choice(len(Xz), min(30000, len(Xz)), replace=False)
    km4 = fit_kmeans(Xz, 4, args.seed)
    algos = {
        "kmeans": km4.labels_[sub],
        "gmm": GaussianMixture(n_components=4, random_state=args.seed).fit(Xz).predict(Xz)[sub],
        "ward": AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(Xz[sub]),
    }
    names = list(algos)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            arows.append({"a": a, "b": b, "ari": float(adjusted_rand_score(algos[a], algos[b]))})
        arows.append({"a": a, "b": "shipped", "ari": float(adjusted_rand_score(algos[a], shipped[sub]))})
    pd.DataFrame(arows).to_csv(out / "algorithms.csv", index=False)

    # ---------- demographic confound ----------
    demo = df.iloc[:len(shipped)].copy() if len(df) != len(shipped) else df.copy()
    demo = demo.assign(_cluster=shipped)
    demo["age_band"] = pd.cut(demo["age"], [0, 18, 22, 26, 35, 50, 120],
                              labels=["<=18", "19-22", "23-26", "27-35", "36-50", "50+"])
    demo["country_top"] = demo["country"].where(
        demo["country"].isin(demo["country"].value_counts().head(10).index), "OTHER")
    drows = []
    for factor in ("sex", "age_band", "country_top"):
        ct = pd.crosstab(demo["_cluster"], demo[factor])
        v = cramers_v(ct)
        p = permutation_p(demo["_cluster"].to_numpy(), demo[factor], v, args.perm, args.seed)
        drows.append({"factor": factor, "cramers_v": v, "perm_p": p, "n_perm": args.perm})
        print(f"  demo {factor}: V={v:.4f} p={p:.4g}", flush=True)
    pd.DataFrame(drows).to_csv(out / "demographics.csv", index=False)

    # ---------- roll-up ----------
    z = ksel[ksel.preprocessing == "zscore"]
    best_sil = int(z.loc[z.silhouette.idxmax(), "k"])
    best_db = int(z.loc[z.davies_bouldin.idxmin(), "k"])
    k4 = z[z.k == 4].iloc[0]
    summary = {
        "n_rows": int(len(df)),
        "k_by_silhouette_zscore": best_sil,
        "k_by_davies_bouldin_zscore": best_db,
        "silhouette_at_k4": float(k4.silhouette),
        "silhouette_best": float(z.silhouette.max()),
        "k4_is_optimal_by_silhouette": best_sil == 4,
        "stability_ari_at_k4_zscore": float(
            stab[(stab.preprocessing == "zscore") & (stab.k == 4)].ari_mean.iloc[0]),
        "demographics": drows,
        "runtime_sec": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== E1 SUMMARY ===")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
