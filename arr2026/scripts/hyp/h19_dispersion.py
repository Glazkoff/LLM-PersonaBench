"""H19: does S_0 actually prefer LOW-dispersion predictors?

App. E asserts a mechanism -- that S_0 ships weak models because their
predictions sit near the constant vector that maximises it. That is a causal
claim and it was never tested. S_0 is maximised by a POINT MASS at the central
value, so it should reward low dispersion; a near-uniform predictor is the
opposite extreme. If the models S_0 picks are high-dispersion, the mechanism in
the paper is backwards and must be rewritten.
"""
import json, glob, os
import numpy as np

def s0(P, Y):
    """S_0 generalised to K levels -- the 5-level version silently dropped
    HEXACO (K=7), the one instrument where the scale length differs."""
    K = P.shape[2]
    lv = np.arange(1, K + 1, dtype=float)
    d = np.abs(lv[None, None, :] - Y[:, :, None]) / (K - 1)
    return np.nanmean(np.where(np.isnan(Y), np.nan, np.nansum(P * (1.0 - d), axis=2)))


def crps(P, Y):
    K = P.shape[2]
    Q = np.cumsum(P, axis=2)[:, :, :K - 1]
    ind = np.stack([(Y <= t).astype(float) for t in range(1, K)], axis=2)
    return np.nanmean(np.where(np.isnan(Y), np.nan,
                               np.nansum((Q - ind) ** 2, axis=2) / (K - 1)))


for corpus in ("ipip", "sd3", "big5", "hexaco"):
    rows = []
    for d in sorted(glob.glob(f"arr2026/results_euler/h17n512_*_{corpus}_val")):
        try:
            meta = json.load(open(os.path.join(d, "summary.json")))
            if not meta.get("valid"):
                continue
            P = np.load(os.path.join(d, "belief_probs.npy"))
            Y = np.load(os.path.join(d, "target_answers.npy"))
        except Exception:
            continue
        # The model answers the LITERAL item; IPIP-family corpora store answers
        # already reverse-recoded. Without this flip every score on a recoded
        # corpus is computed against mis-oriented targets -- the same defect
        # that invalidated an earlier round of this project. h17_selection.py
        # applies it; this script did not.
        if meta.get("corpus_recoded"):
            rk = set(meta.get("reverse_keyed_targets", []))
            flip = np.array([j in rk for j in meta.get("target_ids", [])], dtype=bool)
            if flip.any() and flip.size == P.shape[1]:
                P[:, flip, :] = P[:, flip, ::-1]
        K = P.shape[2]
        lv = np.arange(1, K + 1, dtype=float)
        mean = (P * lv).sum(-1)
        var = (P * (lv ** 2)).sum(-1) - mean ** 2
        ent = -(P * np.log(np.clip(P, 1e-12, None))).sum(-1)
        rows.append(dict(model=os.path.basename(d).replace(f"h17n512_", "").replace(f"_{corpus}_val", ""),
                         sd=float(np.sqrt(np.clip(var, 0, None)).mean()),
                         ent=float(ent.mean()),
                         s0=float(s0(P, Y)),
                         crps=float(crps(P, Y))))
    if len(rows) < 3:
        continue
    sd = np.array([r["sd"] for r in rows]); s0v = np.array([r["s0"] for r in rows])
    cr = np.array([r["crps"] for r in rows]); en = np.array([r["ent"] for r in rows])
    ok = ~np.isnan(s0v)
    print(f"\n=== {corpus} ({len(rows)} models) ===")
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    if ok.sum() >= 3:
        # Report both: the manuscript describes these as rank correlations, so
        # Spearman is the figure it should quote. -cr is the NEGATED CRPS loss,
        # i.e. the proper score itself (higher better), not its negation.
        print(f"  pearson(S0, sd)={np.corrcoef(s0v[ok], sd[ok])[0,1]:+.3f}  "
              f"spearman(S0, sd)={spearman(s0v[ok], sd[ok]):+.3f}")
        print(f"  pearson(S0, entropy)={np.corrcoef(s0v[ok], en[ok])[0,1]:+.3f}  "
              f"spearman(S0, entropy)={spearman(s0v[ok], en[ok]):+.3f}")
        print(f"  pearson(S_1/2 as reward, sd)={np.corrcoef(-cr[ok], sd[ok])[0,1]:+.3f}  "
              f"spearman={spearman(-cr[ok], sd[ok]):+.3f}")
        best_s0 = rows[int(np.nanargmax(s0v))]
        best_cr = rows[int(np.argmin(cr))]
        print(f"  S_0 picks   {best_s0['model']:34s} sd={best_s0['sd']:.3f} ent={best_s0['ent']:.3f}")
        print(f"  CRPS picks  {best_cr['model']:34s} sd={best_cr['sd']:.3f} ent={best_cr['ent']:.3f}")
