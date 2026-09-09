r"""H14: does the persona carry value, and why does it still lose?

The paper has been sliding between two questions -- "does the simulator reproduce
the cluster's response distribution?" and "does it identify the right
respondent?" -- and they have different answers. This separates them.

Every quantity is a normalised CRPS on the 5-level scale, scoring a predicted
CDF against the respondent's own answer:

    L(Q, y) = (1/4) sum_{t=1..4} ( q_t - 1{y <= t} )^2 ,  q_t = Q(X <= t).

Four predictors are compared on the SAME respondents:

    correct    the model's belief for the persona it was given
    permuted   the model's beliefs, shuffled among respondents within a cluster
               -- identical mixture distribution, destroyed identity
    prior      the cluster's empirical item distribution, ignoring the persona
    decoder    a supervised conditional CDF fitted on held-out respondents

Permuting rows leaves every population-level statistic untouched, so any gap
between `correct` and `permuted` is conditioning value that population metrics
cannot see.

We also report the exact decomposition (per item, within cluster)

    E L(prior) - E L(Q) = (1/4) sum_t Var[p_t(C)]  -  (1/4) sum_t E[(q_t - p_t)^2]
                          \_____ value of conditioning _____/   \__ prediction error __/

with p_t the fitted conditional CDF. It states precisely how a simulator can use
the persona in the right direction and still lose to the unconditional
distribution: its prediction error exceeds the value it extracts.
"""
import argparse
import bisect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _orientation import to_human_orientation  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ITEMS = [f"i{i}" for i in range(1, 121)]
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
BINS, L = [0, 20, 40, 60, 80, 100], 5
THRESH = [1, 2, 3, 4]          # P(Y <= t) for t = 1..4


def channel(cl):
    b = ROOT / "src/prompt/mean_value_cluster"
    tr = json.loads((b / "traits.json").read_text())
    fa = json.loads((b / "facets.json").read_text())
    return list(tr.get(str(cl), tr)) + list(fa.get(str(cl), fa))


def q(v):
    return np.clip([bisect.bisect_right(BINS, x) - 1 for x in v], 0, L - 1)


def oh(a, k):
    m = np.zeros((len(a), k * L))
    for j in range(k):
        m[np.arange(len(a)), j * L + a[:, j]] = 1.0
    return m


def ridge(X, Y, Xp, lam=1e-2):
    Xc = np.hstack([X, np.ones((len(X), 1))])
    Xpc = np.hstack([Xp, np.ones((len(Xp), 1))])
    return Xpc @ np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ Y)



def isotonic_rows(D):
    """Project each length-4 threshold vector onto the non-decreasing cone."""
    out = D.copy()
    flat = out.reshape(-1, D.shape[-1])
    for r in range(flat.shape[0]):
        v = flat[r]
        # pool adjacent violators, unit weights
        vals, wts = [], []
        for x in v:
            vals.append(float(x)); wts.append(1.0)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                b, wb = vals.pop(), wts.pop()
                a, wa = vals.pop(), wts.pop()
                vals.append((a * wa + b * wb) / (wa + wb)); wts.append(wa + wb)
        o, i = [], 0
        for val, w in zip(vals, wts):
            o.extend([val] * int(w))
        flat[r] = np.clip(np.array(o[:D.shape[-1]]), 0.0, 1.0)
    return flat.reshape(D.shape)


def crps_from_cdf(Qcdf, Y):
    """Qcdf: (n, items, 4) predicted P(X<=t). Y: (n, items) observed answers."""
    ind = np.stack([(Y <= t).astype(float) for t in THRESH], axis=2)
    # (1/4) * sum over the four thresholds == the mean over them; do not divide twice
    return np.nanmean((Qcdf - ind) ** 2, axis=2)


def probs_to_cdf(P):
    """(n, items, 5) simplex -> (n, items, 4) CDF at t=1..4."""
    return np.cumsum(P, axis=2)[:, :, :4]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n-personas", type=int, default=40)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--seed", type=int, default=260909)
    a = ap.parse_args()

    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")
    rng = np.random.default_rng(a.seed)
    ctx = {}
    for cl in (0, 1, 2, 3):
        sub = df[df.clusters == cl]
        per, fit = sub.iloc[: a.n_personas], sub.iloc[a.n_personas:]
        Yp = per[ITEMS].to_numpy(float)
        Yf = fit[ITEMS].to_numpy(float)
        # cluster prior: empirical CDF per item, from held-out respondents only
        prior = np.stack([(Yf <= t).mean(axis=0) for t in THRESH], axis=1)   # (items,4)
        # supervised conditional CDF on the same code the prompt carries
        c = [x for x in channel(cl) if x in df.columns]
        Xf = oh(np.column_stack([q(fit[x].to_numpy(float)) for x in c]), len(c))
        Xp = oh(np.column_stack([q(per[x].to_numpy(float)) for x in c]), len(c))
        tgt = np.concatenate([(Yf <= t).astype(float) for t in THRESH], axis=1)
        dec = np.clip(ridge(Xf, tgt, Xp), 0, 1).reshape(len(per), len(THRESH), 120)
        dec = np.transpose(dec, (0, 2, 1))                                    # (n,items,4)
        # Independent per-threshold ridges are not monotone in t, so ~2% of
        # predictions implied negative category probabilities. Project each
        # four-vector onto the monotone cone (pool-adjacent-violators).
        dec = isotonic_rows(dec)
        ctx[cl] = {"Y": Yp, "prior": np.broadcast_to(prior, (len(per), 120, 4)),
                   "dec": dec}
        print(f"[cluster {cl}: prior and decoder fitted on {len(fit)} held-out]", flush=True)

    print(f"\n{'run':20s} {'correct':>9s} {'permuted':>9s} {'prior':>9s} {'decoder':>9s} "
          f"{'spread D_Q':>11s} {'mix.exc':>9s} {'residual':>10s}")
    for run in a.runs:
        rd = None
        for base in a.results:
            if (ROOT / base / run / "summary.json").exists():
                rd = ROOT / base / run; break
        if rd is None:
            print(f"{run:20s} MISSING"); continue
        acc = {k: [] for k in ("correct", "permuted", "prior", "dec", "D_Q", "mix_excess")}
        for cl in ctx:
            d = rd / f"readout_cluster_{cl}"
            if not (d / "belief_probs.npy").exists():
                continue
            P = np.load(d / "belief_probs.npy")
            Q = probs_to_cdf(to_human_orientation_cdf(P))
            Y, prior, dec = ctx[cl]["Y"], ctx[cl]["prior"], ctx[cl]["dec"]
            n = min(len(Q), len(Y))
            Q, Y2, prior2, dec2 = Q[:n], Y[:n], prior[:n], dec[:n]
            acc["correct"].append(np.nanmean(crps_from_cdf(Q, Y2)))
            acc["prior"].append(np.nanmean(crps_from_cdf(prior2, Y2)))
            acc["dec"].append(np.nanmean(crps_from_cdf(dec2, Y2)))
            # Exact expectation of the permutation control: average the loss of
            # every donor row against every respondent, which removes Monte Carlo
            # noise from the quantity the headline claim rests on.
            per_pair = np.stack([np.nanmean(crps_from_cdf(
                np.broadcast_to(Q[k:k + 1], Q.shape), Y2), axis=1) for k in range(n)])
            acc["permuted"].append(float(per_pair.mean()))
            acc.setdefault("row_perm", []).append(per_pair.mean(axis=0))
            # per-respondent losses, kept for a paired interval on the gap that
            # matters: does assigning beliefs to the RIGHT person help?
            acc.setdefault("row_correct", []).append(np.nanmean(crps_from_cdf(Q, Y2), axis=1))
            acc.setdefault("row_prior", []).append(np.nanmean(crps_from_cdf(prior2, Y2), axis=1))
            # Exact, observable decomposition -- no oracle, no fitted reference:
            #   R(prior) - R(correct) = I - D_Q - [ R(qbar) - R(prior) ]
            # I is the exact expected permutation gain, D_Q the across-persona
            # dispersion of the model's own CDFs, and the bracket the excess loss
            # of the model's MIXTURE against the prior. Every term is measured.
            qbar = np.nanmean(Q, axis=0, keepdims=True)
            D_Q = float(np.nanmean(np.nanmean((Q - qbar) ** 2, axis=0)))
            R_qbar = float(np.nanmean(crps_from_cdf(np.broadcast_to(qbar, Q.shape), Y2)))
            R_prior = float(np.nanmean(crps_from_cdf(prior2, Y2)))
            acc["D_Q"].append(D_Q)
            acc["mix_excess"].append(R_qbar - R_prior)
        if not acc["correct"]:
            print(f"{run:20s} NO TENSORS"); continue
        m = {k: float(np.mean(v)) for k, v in acc.items()
             if not k.startswith("row_")}
        rc = np.concatenate(acc["row_correct"]); rp = np.concatenate(acc["row_perm"])
        rr = np.concatenate(acc["row_prior"])
        d_id = rp - rc                       # >0 means correct assignment helps
        d_pr = rr - rc                       # >0 means the model beats the prior
        bs_id = np.array([np.mean(rng.choice(d_id, len(d_id))) for _ in range(2000)])
        bs_pr = np.array([np.mean(rng.choice(d_pr, len(d_pr))) for _ in range(2000)])
        ci_id = np.percentile(bs_id, [2.5, 97.5]); ci_pr = np.percentile(bs_pr, [2.5, 97.5])
        ident = m['permuted'] - m['correct']
        resid = (m['prior'] - m['correct']) - (ident - m['D_Q'] - m['mix_excess'])
        print(f"{run:20s} {m['correct']:9.4f} {m['permuted']:9.4f} {m['prior']:9.4f} "
              f"{m['dec']:9.4f} {m['D_Q']:11.4f} {m['mix_excess']:9.4f} {resid:+.2e}",
              flush=True)
        print(f"{'':20s}   identity gain {d_id.mean():+.4f} [{ci_id[0]:+.4f},{ci_id[1]:+.4f}]"
              f" | vs prior {d_pr.mean():+.4f} [{ci_pr[0]:+.4f},{ci_pr[1]:+.4f}]", flush=True)


def to_human_orientation_cdf(P):
    """Reverse-keyed items: flipping the answer reverses the simplex."""
    from _orientation import NEG
    out = P.copy()
    out[:, NEG, :] = out[:, NEG, ::-1]
    return out


if __name__ == "__main__":
    main()
