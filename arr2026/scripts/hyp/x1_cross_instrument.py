"""
X1 -- Does the audit generalise beyond IPIP-NEO-120?

Every result in the paper comes from one instrument. The obvious reviewer
objection is that the whole thing is a property of that questionnaire. This
replicates the two structural findings on an INDEPENDENT instrument with real
human responses: the Dark Triad (SD3) -- 18,192 respondents x 27 items
(Machiavellianism / Narcissism / Psychopathy), a different construct, a different
scale range, a different population.

Two findings are tested, neither of which needs an LLM:

  P1  The Proposition. Is GMD >= MAD on every item, i.e. is a faithful sampler
      dominated by a constant under S? The gap (GMD-MAD)/range is the structural
      penalty. If this fails anywhere, the Proposition is IPIP-specific.

  P2  The human-ceiling inversion. Does the per-item constant beat a real
      respondent drawn from the same population? On IPIP it did, in 20/20 cells.

Also reports how the penalty scales with the number of response options, since
SD3 uses a different scale width than IPIP's 1-5.

CPU only. No LLM calls.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[3]

def gmd(v):
    x = np.sort(np.asarray(v, float)); n = len(x)
    if n < 2: return 0.0
    i = np.arange(1, n + 1)
    return float(2.0 * np.sum((2 * i - n - 1) * x) / (n * n))

def mad_med(v):
    return float(np.mean(np.abs(np.asarray(v, float) - np.median(v))))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="arr2026/results/x1")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    df = pd.read_csv(ROOT / "data/PAlign/Dark-Triad.csv", sep="\t")
    items = [c for c in df.columns if c[0] in "MNP" and c[1:].isdigit()]
    Y = df[items].to_numpy(float)
    Y = Y[~np.isnan(Y).any(axis=1)]
    Y = Y[(Y >= 1).all(axis=1)]          # 0 encodes "not answered" in SD3 dumps
    rng_max, rng_min = Y.max(), Y.min()
    span = rng_max - rng_min             # scale range, replaces IPIP's 4
    print(f"Dark Triad: {Y.shape[0]} complete respondents x {Y.shape[1]} items, "
          f"scale {rng_min:.0f}-{rng_max:.0f} (span {span:.0f})", flush=True)

    # ---- P1: Proposition on every item ----
    mads = np.array([mad_med(Y[:, j]) for j in range(Y.shape[1])])
    gmds = np.array([gmd(Y[:, j]) for j in range(Y.shape[1])])
    holds = int((gmds >= mads - 1e-12).sum())
    s_const = float(1 - mads.mean() / span)
    s_faith = float(1 - gmds.mean() / span)
    print(f"P1  GMD>=MAD on {holds}/{len(items)} items | "
          f"S(const)={s_const:.4f}  S(faithful)={s_faith:.4f}  "
          f"gap={100*(s_const-s_faith):.2f} pp", flush=True)

    # ---- P2: does a constant beat a real respondent? ----
    def sim(pred, human):
        return float(np.mean(1.0 - np.abs(pred - human) / span))
    wins = 0
    const_scores, human_scores = [], []
    for b in range(a.n_boot):
        r = np.random.default_rng(a.seed + b)
        idx = r.choice(len(Y), a.n_eval + 200, replace=False)
        train, test = Y[idx[:200]], Y[idx[200:]]
        const = np.round(train.mean(axis=0))
        cs = float(np.mean([sim(const, t) for t in test]))
        hs = float(np.mean([sim(train[r.integers(len(train))], t) for t in test]))
        const_scores.append(cs); human_scores.append(hs)
        wins += int(cs > hs)
    cs_m, hs_m = float(np.mean(const_scores)), float(np.mean(human_scores))
    print(f"P2  constant={cs_m:.4f}  real human={hs_m:.4f}  "
          f"constant wins {wins}/{a.n_boot} bootstraps", flush=True)

    summary = {
        "instrument": "Dark Triad (SD3)",
        "n_respondents": int(Y.shape[0]), "n_items": int(Y.shape[1]),
        "scale_span": float(span),
        "P1_proposition_holds_items": holds, "P1_items_total": int(Y.shape[1]),
        "P1_S_constant": round(s_const, 4), "P1_S_faithful": round(s_faith, 4),
        "P1_gap_pp": round(100 * (s_const - s_faith), 3),
        "P2_constant_score": round(cs_m, 4), "P2_human_score": round(hs_m, 4),
        "P2_constant_wins": wins, "P2_n_bootstraps": a.n_boot,
        "IPIP_reference": {"gap_pp": 7.75, "constant": 0.7961, "human": 0.7213},
        "verdict": ("GENERALISES: both structural findings replicate on an "
                    "independent instrument and construct"
                    if holds == Y.shape[1] and wins > 0.95 * a.n_boot else
                    "PARTIAL: see per-finding fields"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame({"item": items, "MAD": mads, "GMD": gmds}).to_csv(out / "per_item.csv", index=False)
    print("\n=== X1 SUMMARY ==="); print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
