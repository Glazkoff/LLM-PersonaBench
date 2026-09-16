"""
H22 -- bootstrap confidence intervals for the H4 belief-variance decomposition.

H4 reports, per model, how much of the belief variance sits BETWEEN personas
(individuation) versus WITHIN a persona (response noise), plus the variance
ratio VR against the human across-respondent SD. Those were point estimates
with no uncertainty, so a gap like gemma-4's share_between=0.518 against
gpt-oss's 0.016 could not be separated from a gap like granite's 0.068 against
Qwen's 0.076. This script supplies the intervals.

Resampling unit: the PERSONA. Between-persona variance is a statistic over
personas, so personas are the exchangeable unit; items are held fixed.

Pairing: h4 assigns the SAME personas (df[df.clusters==cl].iloc[:n]) to every
model, so the models are paired by construction. Each bootstrap replicate draws
one set of persona indices per cluster and applies it to ALL models. That makes
the model-vs-model differences a paired bootstrap, which is both correct and
tighter than resampling each model independently.

No GPU is needed: everything is recomputed from the saved belief_probs.npy
tensors (personas x items x 5), which are the exact quantities h4 reduced.

Orientation note: VR is a ratio of SDs and SD is invariant under the
reverse-recoding reflection x -> 6-x, so this measurement is unaffected by the
corpus orientation defect that required fixes in h19/h20d.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ITEMS = [f"i{i}" for i in range(1, 121)]
VALS = np.arange(1, 6).astype(float)


def decompose(probs, human_sd):
    """h4's law-of-total-variance reduction, verbatim in math."""
    ok = human_sd > 0
    # A persona-item cell whose probability vector is entirely missing must stay
    # missing. np.nansum over an all-NaN slice returns 0.0, which would enter the
    # moments as an answer of 0 -- impossible on a 1..5 scale -- and inflate the
    # between-persona variance. Mask those cells back to NaN so the nanmean /
    # nanvar below skip them instead of averaging a fabricated zero.
    missing = np.isnan(probs).all(axis=2)
    mu_ij = np.nansum(probs * VALS[None, None, :], axis=2)
    ex2_ij = np.nansum(probs * (VALS ** 2)[None, None, :], axis=2)
    mu_ij[missing] = np.nan
    ex2_ij[missing] = np.nan
    var_ij = np.clip(ex2_ij - mu_ij ** 2, 0, None)
    within_var = np.nanmean(var_ij, axis=0)
    between_var = np.nanvar(mu_ij, axis=0)
    mixture_sd = np.sqrt(within_var + between_var)
    denom = max(float(np.nansum(within_var[ok] + between_var[ok])), 1e-12)
    # VR_within / VR_between follow h13_individuation_fidelity.py: the per-item
    # component SD divided by the human across-respondent SD, averaged over
    # items with non-degenerate human spread. VR_total is h4's VR_belief_mixture.
    return {
        "share_between": float(np.nansum(between_var[ok]) / denom),
        "VR_within": float(np.nanmean(np.sqrt(within_var[ok]) / human_sd[ok])),
        "VR_between": float(np.nanmean(np.sqrt(between_var[ok]) / human_sd[ok])),
        "VR_mixture": float(np.nanmean(mixture_sd[ok] / human_sd[ok])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="arr2026/results_euler")
    ap.add_argument("--tags", nargs="+", required=True,
                    help="h4r_* directory names, one per model")
    ap.add_argument("--clusters", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="arr2026/results_euler/h22_variance_ci")
    args = ap.parse_args()

    res = ROOT / args.results
    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")

    # Load every model's tensors up front; assert the persona axis matches so
    # the paired resampling is actually valid rather than assumed.
    data, human = {}, {}
    n_personas = {}
    for cl in args.clusters:
        human[cl] = np.nanstd(df[df.clusters == cl][ITEMS].to_numpy(float), axis=0)
    for tag in args.tags:
        data[tag] = {}
        for cl in args.clusters:
            p = res / tag / f"readout_cluster_{cl}" / "belief_probs.npy"
            if not p.exists():
                print(f"MISSING {p}", flush=True)
                sys.exit(3)
            a = np.load(p)
            data[tag][cl] = a
            n_personas.setdefault(cl, a.shape[0])
            if a.shape[0] != n_personas[cl]:
                print(f"PERSONA-AXIS MISMATCH at {tag} cluster {cl}: "
                      f"{a.shape[0]} vs {n_personas[cl]} -- pairing invalid",
                      flush=True)
                sys.exit(3)
    print(f"{len(args.tags)} models x {len(args.clusters)} clusters; "
          f"personas per cluster: {n_personas}", flush=True)

    # Point estimates (cluster-averaged, matching h4's summary).
    point = {}
    for tag in args.tags:
        s = [decompose(data[tag][cl], human[cl]) for cl in args.clusters]
        point[tag] = {k: float(np.mean([x[k] for x in s])) for k in s[0]}

    # Paired bootstrap over personas.
    rng = np.random.default_rng(args.seed)
    STATS = ("share_between", "VR_within", "VR_between", "VR_mixture")
    draws = {tag: {k: [] for k in STATS} for tag in args.tags}
    for b in range(args.n_boot):
        idx = {cl: rng.integers(0, n_personas[cl], n_personas[cl])
               for cl in args.clusters}
        for tag in args.tags:
            s = [decompose(data[tag][cl][idx[cl]], human[cl])
                 for cl in args.clusters]
            for k in STATS:
                draws[tag][k].append(float(np.mean([x[k] for x in s])))
        if (b + 1) % 500 == 0:
            print(f"  bootstrap {b+1}/{args.n_boot}", flush=True)

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    out = {"n_boot": args.n_boot, "seed": args.seed, "models": {}}
    print("\n=== per-model (95% percentile CI over personas) ===", flush=True)
    for tag in args.tags:
        e = {}
        for k in STATS:
            lo, hi = ci(draws[tag][k])
            e[k] = {"point": point[tag][k], "lo": lo, "hi": hi}
            print(f"{tag:34s} {k:14s} {point[tag][k]:.4f}  [{lo:.4f}, {hi:.4f}]",
                  flush=True)
        out["models"][tag] = e

    # Paired pairwise differences on share_between.
    print("\n=== paired differences in VR_between ===", flush=True)
    out["pairs_vrb"] = {}
    for i, a in enumerate(args.tags):
        for bb in args.tags[i + 1:]:
            d = np.array(draws[a]["VR_between"]) - np.array(draws[bb]["VR_between"])
            lo, hi = ci(d)
            pt = point[a]["VR_between"] - point[bb]["VR_between"]
            sep = "SEPARATED" if (lo > 0 or hi < 0) else "overlaps zero"
            print(f"{a} - {bb}: {pt:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {sep}", flush=True)
            out["pairs_vrb"][f"{a}|{bb}"] = {"point": pt, "lo": lo, "hi": hi,
                                             "separated": bool(lo > 0 or hi < 0)}

    print("\n=== paired differences in share_between ===", flush=True)
    out["pairs"] = {}
    for i, a in enumerate(args.tags):
        for bb in args.tags[i + 1:]:
            d = np.array(draws[a]["share_between"]) - np.array(draws[bb]["share_between"])
            lo, hi = ci(d)
            pt = point[a]["share_between"] - point[bb]["share_between"]
            sep = "SEPARATED" if (lo > 0 or hi < 0) else "overlaps zero"
            print(f"{a} - {bb}: {pt:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {sep}", flush=True)
            out["pairs"][f"{a}|{bb}"] = {"point": pt, "lo": lo, "hi": hi,
                                         "separated": bool(lo > 0 or hi < 0)}

    d = ROOT / args.out
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {d/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
