r"""H26: what is an LLM persona worth, measured in the person's own answers?

The screens in H25 report that an LLM persona adds nothing to a conditional
model of the person's observed answers. "Adds nothing" is a true statement and
a useless number. This converts it into a usable one.

The persona prompt for a respondent is built from all 60 observed IPIP answers
(as 30 facet scores). So ask the exchange-rate question directly:

    a decoder given m of that person's actual answers matches the calibrated
    LLM persona at m = ?

m* is the ANSWER-EQUIVALENT of the persona. It is a single interpretable number
with a confidence interval, it is comparable across models, instruments and
elicitations, and it is meaningful whichever way it comes out: m* = 40 would
say persona prompting recovers most of what a questionnaire does, m* = 2 says a
practitioner should collect two more answers instead of running the model.

Everything is selected on a calibration half of the scored panel and applied
frozen to an evaluation half; decoders are fitted on the run's training
respondents. The ridge penalty is retuned at EVERY m -- a decoder held at one
penalty across budgets overfits the small ones and would manufacture a
flattering m* (see arr2026/strengthen/BRIEF.md section 11.2, where exactly that
produced a headline result that did not survive).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h25_incremental_screen import (  # noqa: E402
    ALPHAS, LAMS, boot_ci, iso_fast, onehot, reconstruct, ridge, rps,
    split_items, load_corpus)

BUDGETS = [0, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60]


def fit_decoder(Ztr, targ, Zs, n, nt, K, sel_rows, Yt, CAL):
    lam = min(LAMS, key=lambda l: float(np.nanmean(rps(
        iso_fast(np.transpose(np.clip(ridge(Ztr, targ, Zs, l), 0, 1)
                              .reshape(n, K - 1, nt), (0, 2, 1)))[CAL],
        Yt[CAL], K))))
    flat = np.clip(ridge(Ztr, targ, Zs, lam), 0, 1)
    return iso_fast(np.transpose(flat.reshape(n, K - 1, nt), (0, 2, 1))), lam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=5,
                    help="random observed-item subsets per budget")
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    report = []

    for run_rel in a.runs:
        run = ROOT / run_rel if not Path(run_rel).is_absolute() else Path(run_rel)
        meta = json.loads((run / "summary.json").read_text())
        P = np.load(run / "belief_probs.npy")
        K = int(meta.get("K", P.shape[2]))
        corpus = meta.get("corpus") or meta.get("instrument")
        tgt_ids = meta.get("target_ids")
        recoded = meta.get("corpus_recoded")
        if recoded is None or tgt_ids is None:
            c0 = load_corpus(corpus)
            recoded = c0.recoded if recoded is None else recoded
            tgt_ids = tgt_ids or split_items(c0.ids, c0.scale, c0.n_input)[1]
            meta["target_ids"] = list(tgt_ids)
            if not meta.get("reverse_keyed_targets"):
                meta["reverse_keyed_targets"] = sorted(j for j in tgt_ids if j in c0.rev)
        if recoded:
            rk = set(meta.get("reverse_keyed_targets", []))
            fl = np.array([j in rk for j in meta["target_ids"]], dtype=bool)
            if fl.any():
                P[:, fl, :] = P[:, fl, ::-1]

        Yt = np.load(run / "target_answers.npy")
        Ytr = np.load(run / "train_answers.npy")
        Xs, Xtr, ok = reconstruct(meta, run)
        if not ok:
            print(f"{run.name}: reconstruction mismatch, skipped", flush=True)
            continue
        n, nt = Yt.shape
        n_obs_total = Xs.shape[1]
        th = list(range(1, K))
        Q = np.cumsum(P, axis=2)[:, :, : K - 1]
        sp = np.random.default_rng(20260918).permutation(n)
        CAL, EV = sp[: n // 2], sp[n // 2:]
        prior_row = np.stack([np.nanmean(Ytr <= t, axis=0) for t in th], axis=1)
        targ = np.nan_to_num(
            np.concatenate([(Ytr <= t).astype(float) for t in th], axis=1), nan=0.0)

        # the calibrated LLM: the thing being priced
        Qbar = np.nanmean(Q[CAL], axis=0)
        cal = lambda rows, al: iso_fast(
            np.clip(prior_row[None] + al * (Q[rows] - Qbar[None]), 0, 1))
        alpha = min(ALPHAS, key=lambda al: float(np.nanmean(rps(cal(CAL, al), Yt[CAL], K))))
        llm = float(np.nanmean(rps(cal(EV, alpha), Yt[EV], K)))
        llm_raw = float(np.nanmean(rps(Q[EV], Yt[EV], K)))

        curve = []
        for m in [b for b in BUDGETS if b <= n_obs_total]:
            vals = []
            for rep in range(1 if m in (0, n_obs_total) else a.reps):
                if m == 0:
                    D = np.broadcast_to(prior_row[None], (n, nt, K - 1))
                    lam = None
                else:
                    cols = rng.choice(n_obs_total, m, replace=False)
                    D, lam = fit_decoder(onehot(Xtr[:, cols], K), targ,
                                         onehot(Xs[:, cols], K), n, nt, K,
                                         None, Yt, CAL)
                vals.append(np.nanmean(rps(D[EV], Yt[EV], K), axis=1))
            per = np.mean(np.stack(vals), axis=0)
            curve.append(dict(m=m, loss=float(np.nanmean(per)), lam=lam,
                              per_respondent=per))
            print(f"  {run.name[:40]:40s} m={m:3d}  {np.nanmean(per):.4f}", flush=True)

        # m* = smallest budget whose decoder already matches the calibrated LLM.
        # Reported two ways. The integer crossing is what a practitioner acts on
        # ("ask one more question"). But the crossing is coarse: on IPIP the
        # calibrated LLM sits at 0.1534 between m=0 (0.1538) and m=1 (0.1509),
        # far nearer the floor, so "1" overstates it by up to a whole answer.
        # The interpolated value places the LLM on the curve between the
        # bracketing budgets and is the honest effect size.
        mstar = next((c["m"] for c in curve if c["loss"] <= llm), None)
        mstar_interp = None
        for lo_, hi_ in zip(curve, curve[1:]):
            if lo_["loss"] >= llm >= hi_["loss"] and lo_["loss"] > hi_["loss"]:
                frac = (lo_["loss"] - llm) / (lo_["loss"] - hi_["loss"])
                mstar_interp = float(lo_["m"] + frac * (hi_["m"] - lo_["m"]))
                break
        if mstar_interp is None and curve and llm >= curve[0]["loss"]:
            mstar_interp = 0.0
        hit = next((c for c in curve if c["m"] == mstar), None)
        gap_ci = (np.nan, np.nan)
        if hit is not None:
            d = np.nanmean(rps(cal(EV, alpha), Yt[EV], K), axis=1) - hit["per_respondent"]
            gap_ci = boot_ci(d[np.isfinite(d)], rng)
        report.append(dict(
            run=run.name, model=meta.get("model"), corpus=corpus, K=K,
            n_observed_available=int(n_obs_total), alpha=float(alpha),
            llm_raw=llm_raw, llm_calibrated=llm,
            answer_equivalent=mstar,
            answer_equivalent_interp=mstar_interp,
            answer_equivalent_gap=None if hit is None else float(llm - hit["loss"]),
            gap_ci_lo=float(gap_ci[0]), gap_ci_hi=float(gap_ci[1]),
            curve=[{k: v for k, v in c.items() if k != "per_respondent"} for c in curve]))
        mi = "n/a" if mstar_interp is None else f"{mstar_interp:.2f}"
        print(f"{run.name}: calibrated LLM {llm:.4f} (raw {llm_raw:.4f}, alpha {alpha}) "
              f"-> ANSWER-EQUIVALENT m* = {mstar} (interpolated {mi}) "
              f"of {n_obs_total} observed answers\n", flush=True)

    out = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    for r in report:
        mi = r.get("answer_equivalent_interp")
        print(f"  {r['run'][:46]:46s} m* = {r['answer_equivalent']} "
              f"(interp {'n/a' if mi is None else format(mi, '.2f')}) "
              f"/ {r['n_observed_available']}")


if __name__ == "__main__":
    main()
