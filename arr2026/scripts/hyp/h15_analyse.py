r"""H15 analysis: does the H14 finding survive a disjoint conditioning/target split?

Same four predictors as H14 -- correct assignment, within-sample permutation,
the empirical prior, and a supervised conditional decoder -- but now the persona
was built from items the readout never covers, and every baseline is fitted on
training respondents whose target answers the model never saw.

If the H14 pattern replicates here it is not an artefact of scoring a model on
the same items its conditioning was computed from.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
THRESH = [1, 2, 3, 4]


def cdf_from_probs(P):
    return np.cumsum(P, axis=2)[:, :, :4]


def crps(Q, Y):
    ind = np.stack([(Y <= t).astype(float) for t in THRESH], axis=2)
    return np.nanmean((Q - ind) ** 2, axis=2)


def isotonic_rows(D):
    out = D.copy(); flat = out.reshape(-1, D.shape[-1])
    for r in range(flat.shape[0]):
        vals, wts = [], []
        for x in flat[r]:
            vals.append(float(x)); wts.append(1.0)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                b, wb = vals.pop(), wts.pop(); a_, wa = vals.pop(), wts.pop()
                vals.append((a_ * wa + b * wb) / (wa + wb)); wts.append(wa + wb)
        o = []
        for v, w in zip(vals, wts):
            o.extend([v] * int(w))
        flat[r] = np.clip(np.array(o[:D.shape[-1]]), 0, 1)
    return flat.reshape(D.shape)


def ridge(X, Y, Xp, lam=1e-2):
    Xc = np.hstack([X, np.ones((len(X), 1))]); Xpc = np.hstack([Xp, np.ones((len(Xp), 1))])
    return Xpc @ np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ Y)


def onehot(S, levels=5):
    b = np.clip((S / 20.0).astype(int), 0, levels - 1)
    m = np.zeros((len(S), S.shape[1] * levels))
    for k in range(S.shape[1]):
        m[np.arange(len(S)), k * levels + b[:, k]] = 1.0
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=260909)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    print(f"{'run':28s} {'correct':>8s} {'permuted':>9s} {'prior':>8s} {'decoder':>8s} "
          f"{'identity gain':>22s} {'vs prior':>22s}")
    for run in a.runs:
        d = ROOT / run
        if not (d / "belief_probs.npy").exists():
            print(f"{run:28s} MISSING"); continue
        meta = json.loads((d / "summary.json").read_text())
        if not meta.get("valid", False):
            print(f"{run:28s} INVALID (mass {meta.get('mass_on_scale_mean')})"); continue
        P = np.load(d / "belief_probs.npy")
        Yt = np.load(d / "target_answers.npy")
        Ytr = np.load(d / "train_answers.npy")
        Ste = np.load(d / "code_test.npy"); Str = np.load(d / "code_train.npy")
        Q = cdf_from_probs(P)
        n = len(Q)

        prior = np.stack([(Ytr <= t).mean(axis=0) for t in THRESH], axis=1)
        prior = np.broadcast_to(prior, Q.shape)
        tgt = np.concatenate([(Ytr <= t).astype(float) for t in THRESH], axis=1)
        dec = np.clip(ridge(onehot(Str), tgt, onehot(Ste)), 0, 1)
        dec = isotonic_rows(np.transpose(dec.reshape(n, len(THRESH), Yt.shape[1]), (0, 2, 1)))

        r_cor = np.nanmean(crps(Q, Yt), axis=1)
        r_pri = np.nanmean(crps(prior, Yt), axis=1)
        r_dec = np.nanmean(crps(dec, Yt), axis=1)
        pair = np.stack([np.nanmean(crps(np.broadcast_to(Q[k:k+1], Q.shape), Yt), axis=1)
                         for k in range(n)])
        r_per = pair.mean(axis=0)

        d_id, d_pr = r_per - r_cor, r_pri - r_cor
        bs = lambda v: np.percentile([np.mean(rng.choice(v, len(v))) for _ in range(2000)],
                                     [2.5, 97.5])
        ci_id, ci_pr = bs(d_id), bs(d_pr)
        print(f"{run.split('/')[-1]:28s} {r_cor.mean():8.4f} {r_per.mean():9.4f} "
              f"{r_pri.mean():8.4f} {r_dec.mean():8.4f} "
              f"{d_id.mean():+8.4f} [{ci_id[0]:+.4f},{ci_id[1]:+.4f}] "
              f"{d_pr.mean():+8.4f} [{ci_pr[0]:+.4f},{ci_pr[1]:+.4f}]", flush=True)


if __name__ == "__main__":
    main()
