"""
Recompute the variance fields of an h4 refresh summary from its saved tensors.

h4_variance_ladder_hf.py originally passed missing probability vectors through
np.nansum, which returns 0.0 for an all-NaN slice and therefore entered a
missing persona-item readout as an answer of 0. The script is fixed, but a run
that predates the fix left summary.json and variance_ladder.csv holding the
contaminated aggregates. Re-running the model would need a GPU; the reduction
does not -- the belief tensors it consumed are already on disk. This recomputes
exactly the fields the bug touched and leaves every other field alone.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ITEMS = [f"i{i}" for i in range(1, 121)]
VALS = np.arange(1, 6).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="h4 result dir to repair")
    ap.add_argument("--clusters", type=int, nargs="+", default=[0, 1, 2, 3])
    a = ap.parse_args()

    d = ROOT / a.dir
    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")
    csv = pd.read_csv(d / "variance_ladder.csv")

    changed = []
    for cl in a.clusters:
        probs = np.load(d / f"readout_cluster_{cl}" / "belief_probs.npy")
        human_sd = np.nanstd(df[df.clusters == cl][ITEMS].to_numpy(float), axis=0)
        ok = human_sd > 0

        missing = np.isnan(probs).all(axis=2)
        mu = np.nansum(probs * VALS[None, None, :], axis=2)
        ex2 = np.nansum(probs * (VALS ** 2)[None, None, :], axis=2)
        mu[missing] = np.nan
        ex2[missing] = np.nan
        var_ij = np.clip(ex2 - mu ** 2, 0, None)

        within = np.nanmean(var_ij, axis=0)
        between = np.nanvar(mu, axis=0)
        mix = np.sqrt(within + between)
        denom = max(float(np.nansum(within[ok] + between[ok])), 1e-12)

        row = csv.cluster == cl
        old = float(csv.loc[row, "VR_belief_mixture"].iloc[0])
        new = float(np.nanmean(mix[ok] / human_sd[ok]))
        csv.loc[row, "VR_belief_mixture"] = new
        csv.loc[row, "share_within"] = float(np.nansum(within[ok]) / denom)
        csv.loc[row, "share_between"] = float(np.nansum(between[ok]) / denom)
        csv.loc[row, "coverage"] = float(1.0 - missing.mean())
        if abs(old - new) > 1e-9:
            changed.append((cl, old, new, int(missing.sum())))

    csv.to_csv(d / "variance_ladder.csv", index=False)

    s = json.loads((d / "summary.json").read_text())
    s["VR_belief_mixture_mean"] = round(float(csv.VR_belief_mixture.mean()), 4)
    s["share_within_mean"] = round(float(csv.share_within.mean()), 4)
    s["share_between_mean"] = round(float(csv.share_between.mean()), 4)
    s["coverage"] = round(float(csv.coverage.mean()), 6)
    s["repaired"] = ("variance fields recomputed from the saved tensors after the "
                     "missing-readout fix; fields the bug did not touch are unchanged")
    (d / "summary.json").write_text(json.dumps(s, indent=2))

    for cl, old, new, nmiss in changed:
        print(f"  cluster {cl}: VR_belief_mixture {old:.4f} -> {new:.4f} "
              f"({nmiss} missing cells)")
    if not changed:
        print("  no cluster changed (no missing cells)")
    print(f"VR_belief_mixture_mean -> {s['VR_belief_mixture_mean']}  "
          f"share_between_mean -> {s['share_between_mean']}")


if __name__ == "__main__":
    main()
