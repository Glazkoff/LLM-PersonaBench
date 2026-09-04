"""
H8 -- What is a persona worth, in real respondents?

"The simulator loses to a constant" invites the reply that the constant is a
strawman. This restates the same finding in a unit nobody can wave away: how many
real survey respondents you would have to sample before a plug-in estimate of the
cluster's response distribution matches what the persona gives you.

For each m, draw m real respondents, form the per-item empirical distribution,
sample a synthetic population from it, and score it with the same metrics. Then
invert the curve to read off m* for each audited system.

Per-item categorical distributions over five options are cheap to estimate, so we
expect m* to be small. If it is single digits, the honest framing of the whole
enterprise changes: a persona is not a substitute for a survey, it is a substitute
for a survey of about that many people.

CPU only. No LLM calls.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

ITEMS = [f"i{i}" for i in range(1, 121)]
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "arr2026/results/h8"
MS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 100, 200, 500]
N_SYNTH, N_BOOT = 200, 40


def df_metric(a, b):
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
        for m in MS:
            dfs, vrs = [], []
            for _ in range(N_BOOT):
                idx = rng.choice(len(Y), m, replace=False)
                donor = Y[idx]
                ref = Y[rng.choice(len(Y), 200, replace=False)]
                synth = np.empty((N_SYNTH, 120))
                for j in range(120):
                    col = donor[:, j][~np.isnan(donor[:, j])]
                    synth[:, j] = rng.choice(col if len(col) else [3], N_SYNTH)
                dfs.append(df_metric(synth, ref))
                ok = np.nanstd(ref, 0) > 0
                vrs.append(float(np.nanmean(np.nanstd(synth, 0)[ok] / np.nanstd(ref, 0)[ok])))
            rows.append({"cluster": int(cl), "m": m,
                         "DF_mean": float(np.mean(dfs)), "DF_sd": float(np.std(dfs)),
                         "VR_mean": float(np.mean(vrs))})
            print(f"  cluster {cl} m={m:4d}: DF={rows[-1]['DF_mean']:.4f} "
                  f"VR={rows[-1]['VR_mean']:.3f}", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(OUT / "respondent_equivalents.csv", index=False)

    # invert: smallest m whose DF matches or beats each reference system
    e3 = ROOT / "arr2026/results/e3/metric_suite.csv"
    targets = {}
    if e3.exists():
        a = pd.read_csv(e3); a = a[a.stage == "after"]
        targets["LLM_persona"] = float(a.df_model.mean())
        targets["constant"] = float(a.df_constant_floor.mean())
        targets["human_ceiling"] = float(a.df_human_ceiling.mean())
    curve = t.groupby("m").DF_mean.mean()
    equiv = {}
    for name, tgt in targets.items():
        hit = curve[curve >= tgt]
        equiv[name] = int(hit.index[0]) if len(hit) else None

    summary = {"m_grid": MS, "DF_by_m": curve.round(4).to_dict(),
               "targets": {k: round(v, 4) for k, v in targets.items()},
               "respondent_equivalents": equiv,
               "interpretation": (
                   "respondent_equivalents[LLM_persona] is how many real respondents you "
                   "would need to sample before a plug-in estimate matches the persona. "
                   "A single-digit value restates the audit in units of real people.")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== H8 SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
