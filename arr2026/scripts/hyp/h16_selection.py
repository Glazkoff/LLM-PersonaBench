r"""H16: does the proposed evaluation change which simulator you would ship?

The audit shows the metric is wrong. That is a diagnosis. This asks whether
adopting the replacement changes a decision: given one fixed candidate set,
select under the audited score S_0 and under the proposed S_1/2, then compare
the two picks on respondents no selection decision has seen.

Design constraints, fixed before the validation forecasts were read:

  * Candidates are the eleven belief models, re-run under the disjoint-item
    prompts of App. B. The paper's own h4 readouts CANNOT be used: they were
    conditioned on scores derived from the answers being scored.
  * Baselines are fitted on the training block alone. No cluster label is used
    anywhere -- the released k=4 partition is derived from all 120 items.
  * Selection reads probability forecasts, never greedy answers, so both rules
    see the same object.
  * The scored panel for selection ('val') and for evaluation ('test2') are
    disjoint from each other, from the training block, and from App. B's 256.

Selection rules, both computed from the same forecast P:
  A  S_0     expected answer similarity, E[1 - |X-y|/4] under the forecast
  B  S_1/2   normalised CRPS, the unique proper member of the S_lambda family

Outcome on the fresh panel: test CRPS (primary -- ordinal, penalises a
four-category error more than a one-category error) with a paired
respondent-level bootstrap, categorical log-loss secondary, and the gap to the
training-fitted empirical prior alongside both.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
THRESH = [1, 2, 3, 4]
LEVELS = np.arange(1, 6, dtype=float)


def load(run):
    d = ROOT / run
    meta = json.loads((d / "summary.json").read_text())
    P = np.load(d / "belief_probs.npy")
    if meta.get("corpus_recoded"):
        rk = set(meta["reverse_keyed_targets"])
        flip = np.array([j in rk for j in meta["target_ids"]], dtype=bool)
        if flip.any():
            P[:, flip, :] = P[:, flip, ::-1]
    return meta, P, (np.load(d / "target_answers.npy"),
                     np.load(d / "train_answers.npy"))


def crps(Q, Y):
    """Normalised CRPS on the 5-level scale: (1/4) sum_t (q_t - 1{y<=t})^2."""
    ind = np.stack([(Y <= t).astype(float) for t in THRESH], axis=2)
    return np.where(np.isnan(Y), np.nan, np.nansum((Q - ind) ** 2, axis=2) / 4.0)


def s0(P, Y):
    """Expected answer similarity under the forecast -- the audited metric."""
    d = np.abs(LEVELS[None, None, :] - Y[:, :, None]) / 4.0
    return np.where(np.isnan(Y), np.nan, np.nansum(P * (1.0 - d), axis=2))


def logloss(P, Y):
    idx = np.clip(np.nan_to_num(Y, nan=1).astype(int) - 1, 0, 4)
    p = np.take_along_axis(P, idx[:, :, None], axis=2)[:, :, 0]
    return np.where(np.isnan(Y), np.nan, -np.log(np.clip(p, 1e-12, None)))


def per_respondent(f, arg, Y):
    return np.nanmean(f(arg, Y), axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", nargs="+", required=True,
                    help="validation-panel runs, one per candidate")
    ap.add_argument("--test", nargs="*", default=[],
                    help="test-panel runs for the SELECTED candidates only")
    ap.add_argument("--seed", type=int, default=261009)
    ap.add_argument("--boot", type=int, default=5000)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    # ---- selection, on the validation panel only -------------------------
    print("=== validation panel: both rules see the same forecasts ===")
    print(f"{'candidate':22s} {'S_0 (up)':>10s} {'CRPS (down)':>12s} {'logloss':>9s}")
    rows = {}
    for run in a.val:
        meta, P, (Yt, Ytr) = load(run)
        if not meta.get("valid", False):
            print(f"{Path(run).name:22s} INVALID (mass {meta.get('mass_on_scale_mean')})")
            continue
        Q = np.cumsum(P, axis=2)[:, :, :4]
        rows[Path(run).name] = dict(
            s0=np.nanmean(s0(P, Yt)), crps=np.nanmean(crps(Q, Yt)),
            ll=np.nanmean(logloss(P, Yt)), panel=meta.get("panel"))
        r = rows[Path(run).name]
        print(f"{Path(run).name:22s} {r['s0']:10.4f} {r['crps']:12.4f} {r['ll']:9.4f}")

    if not rows:
        raise SystemExit("no valid validation runs")
    panels = {r["panel"] for r in rows.values()}
    assert panels == {"val"}, f"selection must read the val panel, saw {panels}"

    pick_a = max(rows, key=lambda k: rows[k]["s0"])     # audited metric
    pick_b = min(rows, key=lambda k: rows[k]["crps"])   # proposed metric
    print(f"\nrule A (S_0)   selects {pick_a}")
    print(f"rule B (S_1/2) selects {pick_b}")
    print("AGREE -- no selection difference in this candidate set and split"
          if pick_a == pick_b else "DISAGREE -- the two rules would ship different models")

    if not a.test:
        print("\n(no test-panel runs supplied; run the selected candidates on "
              "--panel test2 and re-invoke with --test)")
        return

    # ---- evaluation, on respondents no selection decision has seen -------
    print("\n=== test panel ===")
    out, prior_scores = {}, None
    for run in a.test:
        meta, P, (Yt, Ytr) = load(run)
        assert meta.get("panel") == "test2", f"{run} is panel {meta.get('panel')}"
        Q = np.cumsum(P, axis=2)[:, :, :4]
        out[Path(run).name] = dict(crps=per_respondent(crps, Q, Yt),
                                   ll=per_respondent(logloss, P, Yt))
        if prior_scores is None:
            pr = np.stack([(Ytr <= t).mean(axis=0) for t in THRESH], axis=1)
            pr = np.broadcast_to(pr, Q.shape)
            pp = np.diff(np.concatenate(
                [np.zeros((*pr.shape[:2], 1)), pr, np.ones((*pr.shape[:2], 1))], axis=2), axis=2)
            prior_scores = dict(crps=per_respondent(crps, pr, Yt),
                                ll=per_respondent(logloss, pp, Yt))

    for name, v in list(out.items()) + [("empirical prior", prior_scores)]:
        print(f"{name:22s} CRPS {v['crps'].mean():.4f}   logloss {v['ll'].mean():.4f}")

    names = list(out)
    if len(names) == 2:
        for key, lab in (("crps", "CRPS"), ("ll", "log-loss")):
            d = out[names[0]][key] - out[names[1]][key]
            bs = np.percentile([d[rng.integers(0, len(d), len(d))].mean()
                                for _ in range(a.boot)], [2.5, 97.5])
            print(f"paired {lab:9s} {names[0]} - {names[1]}: "
                  f"{d.mean():+.4f} [{bs[0]:+.4f},{bs[1]:+.4f}]  "
                  f"{'resolved' if bs[0]*bs[1] > 0 else 'unresolved'}")
    else:
        print("\nBoth rules selected the same model, so there is no paired "
              "difference to report. The supported statement is that the two "
              "rules did not differ on this candidate set and split -- not that "
              "the ranking they induce is right.")


if __name__ == "__main__":
    main()
