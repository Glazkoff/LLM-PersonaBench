"""
H10 -- Is bootstrap stability evidence of structure at all?

The paper reports that k=4 has silhouette 0.082 (no separation) yet bootstrap
ARI 0.932 (high stability), and resolves the tension by saying the two measure
different things. That is asserted, not tested. A reviewer can reasonably ask
whether ARI 0.93 is impressive or is what ANY smooth unimodal cloud in 30
dimensions would give at n=410,168.

This calibrates the stability number against structureless surrogates:

  gauss      multivariate normal matched to the real facet covariance
  copula     Gaussian copula matched to per-facet empirical marginals AND
             covariance (preserves the skew and bounded support of facet scores)
  gmm4       POSITIVE CONTROL: a genuinely 4-component well-separated mixture

If the structureless surrogates also reach ARI ~0.93 at silhouette ~0.08, then
stability is not discriminative here and the paper should say so plainly. If they
sit far lower, the k=4 partition is better supported than silhouette suggests and
the paper's current hedge can be strengthened into a defence.

CPU only.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata, norm

FACETS = [c for c in [
    "facet_imagination","facet_artistic_interests","facet_emotionality","facet_adventurousness",
    "facet_intellect","facet_liberalism","facet_self_efficacy","facet_orderliness",
    "facet_dutifulness","facet_achievement_striving","facet_self_discipline","facet_cautiousness",
    "facet_friendliness","facet_gregariousness","facet_assertiveness","facet_activity_level",
    "facet_excitement_seeking","facet_cheerfulness","facet_trust","facet_morality","facet_altruism",
    "facet_cooperation","facet_modesty","facet_sympathy","facet_anxiety","facet_anger",
    "facet_depression","facet_self_consciousness","facet_immoderation","facet_vulnerability"]]

def stability(X, k, boot, sub, seed):
    aris = []
    ref = np.random.default_rng(seed).choice(len(X), min(20000, len(X)), replace=False)
    for b in range(boot):
        r = np.random.default_rng(seed + b)
        i1 = r.choice(len(X), sub, replace=True); i2 = r.choice(len(X), sub, replace=True)
        m1 = KMeans(k, n_init=10, random_state=seed + b).fit(X[i1])
        m2 = KMeans(k, n_init=10, random_state=seed + 900 + b).fit(X[i2])
        aris.append(adjusted_rand_score(m1.predict(X[ref]), m2.predict(X[ref])))
    return float(np.mean(aris)), float(np.std(aris))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/df_ipipneo_120_clusters")
    ap.add_argument("--out", default="arr2026/results/h10")
    ap.add_argument("--boot", type=int, default=30)
    ap.add_argument("--sub", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.data)
    X = StandardScaler().fit_transform(df[FACETS].to_numpy(float))
    n, d = X.shape
    rng = np.random.default_rng(a.seed)
    cov = np.cov(X, rowvar=False)

    surr = {"real": X}
    surr["gauss"] = rng.multivariate_normal(np.zeros(d), cov, size=n)
    # Gaussian copula: preserve per-facet marginals exactly, covariance approximately
    Z = rng.multivariate_normal(np.zeros(d), cov, size=n)
    cop = np.empty_like(Z)
    for j in range(d):
        u = norm.cdf((Z[:, j] - Z[:, j].mean()) / (Z[:, j].std() + 1e-12))
        cop[:, j] = np.sort(X[:, j])[np.clip((u * (n - 1)).astype(int), 0, n - 1)]
    surr["copula"] = cop
    # positive control: genuinely separated 4-component mixture
    centres = rng.normal(0, 3.0, size=(4, d))
    lab = rng.integers(0, 4, size=n)
    surr["gmm4_separated"] = centres[lab] + rng.normal(0, 1.0, size=(n, d))

    rows = []
    for name, D in surr.items():
        sidx = rng.choice(len(D), 20000, replace=False)
        km = KMeans(4, n_init=10, random_state=a.seed).fit(D)
        sil = float(silhouette_score(D[sidx], km.labels_[sidx]))
        m, s = stability(D, 4, a.boot, a.sub, a.seed)
        rows.append({"dataset": name, "silhouette_k4": sil, "ari_k4_mean": m, "ari_k4_sd": s})
        print(f"  {name:<18} silhouette={sil:.4f}  bootstrap ARI={m:.3f} +-{s:.3f}", flush=True)

    t = pd.DataFrame(rows); t.to_csv(out / "stability_null.csv", index=False)
    real = t[t.dataset == "real"].iloc[0]
    null_max = t[t.dataset.isin(["gauss", "copula"])].ari_k4_mean.max()
    summary = {
        "real_silhouette_k4": round(float(real.silhouette_k4), 4),
        "real_ari_k4": round(float(real.ari_k4_mean), 4),
        "structureless_max_ari_k4": round(float(null_max), 4),
        "separated_control_ari_k4": round(float(t[t.dataset == "gmm4_separated"].ari_k4_mean.iloc[0]), 4),
        "verdict": ("STABILITY NOT DISCRIMINATIVE: structureless data reaches comparable ARI, "
                    "so the k=4 stability number is not evidence of real clusters"
                    if null_max >= float(real.ari_k4_mean) - 0.05 else
                    "STABILITY IS DISCRIMINATIVE: the real partition is markedly more stable "
                    "than covariance-matched structureless data"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== H10 SUMMARY ==="); print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
