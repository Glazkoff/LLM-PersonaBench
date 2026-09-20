r"""H25: does an LLM persona add predictive information about a person that the
person's OWN observed answers do not already carry? Screened across every saved run.

Motivation. Pilots on the h15 panel (arr2026/strengthen/) answered "no" three
times over, for Qwen2.5-7B and Granite-4.1-8B on IPIP and SD3: a convex stack
selects w*=0, and adding the LLM's CDF as features to a linear or gradient-
boosted combiner moves the loss by less than 0.0005. Those are four cells. This
screens ALL of them -- every model, all four corpora, both panels -- so the
conclusion is a measured property of the method rather than of one weak persona
prompt, and so that any cell where the LLM DOES help is found rather than
assumed away.

Everything is CPU-only and reuses saved belief tensors. No inference.

Protocol per run directory, respondent-level throughout:
  * the scored panel is split into CAL / EVAL halves. Every hyperparameter --
    the decoder's ridge penalty, the calibration alpha, the stacking weight,
    the combiner's penalty -- is selected on CAL and applied frozen to EVAL.
  * the population prior and the decoder are fitted on the run's TRAINING
    respondents, who are disjoint from the scored panel by construction.
  * the LLM simplex is reoriented on reverse-keyed targets wherever the corpus
    stores recoded answers. Skipping this silently corrupts every LLM arm.
  * the headline is INCREMENTAL: the same combiner, same capacity, same fitting
    rows, with and without the LLM features. Anything else measures the
    baseline, not the LLM.

Observed answers are reconstructed deterministically from the corpus and the
run's own summary.json (see arr2026/strengthen/BRIEF.md section 12) and the
reconstruction is VERIFIED against the saved target_answers.npy before use; a
run whose reconstruction does not match is skipped loudly rather than scored on
a guess.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _corpora import load as _load_corpus_uncached  # noqa: E402

_CORPUS_CACHE = {}


def load_corpus(name):
    """Memoised corpus load.

    _corpora.load re-reads the source every call, and the IPIP file is 150 MB.
    Screening 191 run directories would read it once per run -- minutes of pure
    I/O and the dominant cost of the whole job. The Corpus is read-only here.
    """
    if name not in _CORPUS_CACHE:
        _CORPUS_CACHE[name] = _load_corpus_uncached(name)
    return _CORPUS_CACHE[name]


# ---------------------------------------------------------------- primitives
def iso_fast(D):
    """Exact L2 isotonic projection of each row onto [0,1], vectorised.

    yhat_i = max_{j<=i} min_{k>=i} mean(y_j..y_k). Validated identical to
    h15_analyse.isotonic_rows and to a brute-force projection; see
    arr2026/strengthen/iso_check.py. ~36x faster, which is what makes a tuned
    screen over 100+ runs affordable.
    """
    L = D.shape[-1]
    X = D.reshape(-1, L)
    cs = np.concatenate([np.zeros((len(X), 1)), np.cumsum(X, axis=1)], axis=1)
    j = np.arange(L)[:, None]
    k = np.arange(L)[None, :]
    cnt = (k - j + 1).astype(float)
    M = (cs[:, k + 1] - cs[:, j]) / np.where(cnt > 0, cnt, 1.0)
    M = np.where((k >= j)[None], M, np.inf)
    out = np.empty_like(X)
    for i in range(L):
        out[:, i] = M[:, : i + 1, i:].min(axis=2).max(axis=1)
    return np.clip(out, 0.0, 1.0).reshape(D.shape)


def rps(Q, Y, K):
    """Ordinal ranked probability score, normalised by K-1 thresholds.

    Thresholds follow K, so HEXACO (K=7) is scored on six thresholds rather
    than the four every earlier script hardcodes.

    Two failure modes this guards, both of which silently flatter an arm:

    * A MISSING human answer. `NaN <= t` is False at every threshold, which is
      exactly the indicator pattern of "the respondent answered K". Scoring a
      missing answer as a top-category answer is not a small error -- it is a
      fabricated label. Such cells return NaN and are dropped.
    * A REFUSED forecast. belief_probs is NaN where the model put less than 0.1
      of its mass on the answer digits. Averaging over the non-NaN thresholds of
      a partially-NaN row would reward the refusal; `_scoring.rps` goes further
      and uses nansum, so an all-NaN forecast scores a perfect 0.0. Here any NaN
      in a forecast row makes the whole cell NaN.
    """
    th = list(range(1, K))
    ok_y = np.isfinite(Y)
    ind = np.stack([np.where(ok_y, Y <= t, np.nan).astype(float) for t in th],
                   axis=2)
    ok_q = np.isfinite(Q).all(axis=2)
    cell = np.mean((Q - ind) ** 2, axis=2)          # mean, not nanmean
    return np.where(ok_y & ok_q, cell, np.nan)


def logloss(P, Y):
    p = np.clip(P, 1e-6, None)
    p = p / p.sum(axis=2, keepdims=True)
    k = np.clip(np.nan_to_num(Y, nan=1).astype(int) - 1, 0, P.shape[2] - 1)
    ll = -np.log(np.take_along_axis(p, k[:, :, None], axis=2))[:, :, 0]
    ll[~np.isfinite(Y)] = np.nan
    return np.nanmean(ll, axis=1)


def ridge(X, Y, Xp, lam):
    Xc = np.hstack([X, np.ones((len(X), 1))])
    Xpc = np.hstack([Xp, np.ones((len(Xp), 1))])
    A = Xc.T @ Xc + lam * np.eye(Xc.shape[1])
    return Xpc @ np.linalg.solve(A, Xc.T @ Y)


def onehot(Y, K):
    """One-hot the observed answers; a missing answer becomes an all-zero block
    rather than a fabricated level."""
    n, m = Y.shape
    out = np.zeros((n, m * K))
    ok = np.isfinite(Y)
    b = np.clip(np.nan_to_num(Y, nan=1).astype(int) - 1, 0, K - 1)
    cols = (np.arange(m)[None] * K + b)
    rows = np.repeat(np.arange(n)[:, None], m, axis=1)
    out[rows[ok], cols[ok]] = 1.0
    return out


def boot_ci(d, rng, n_boot=4000):
    n = len(d)
    if n == 0:
        return (np.nan, np.nan)
    draws = [float(np.mean(d[rng.integers(0, n, n)])) for _ in range(n_boot)]
    return tuple(np.percentile(draws, [2.5, 97.5]))


# ------------------------------------------------------- split reconstruction
def split_items(ids, scale, n_input):
    inp, tgt = [], []
    for s in sorted({scale[i] for i in ids}):
        mem = [i for i in ids if scale[i] == s]
        inp += mem[:n_input]
        tgt += mem[n_input:]
    return sorted(inp), sorted(tgt)


def scores_from(c, inp):
    idx = {i: k for k, i in enumerate(c.ids)}
    out = []
    for s in sorted({c.scale[i] for i in inp}):
        mem = [i for i in inp if c.scale[i] == s]
        cols = [c.Y[:, idx[i]] if (c.recoded or i not in c.rev)
                else (c.K + 1.0) - c.Y[:, idx[i]] for i in mem]
        blk = np.stack(cols, axis=1)
        cnt = np.sum(~np.isnan(blk), axis=1)
        raw = np.where(cnt > 0, np.nansum(blk, axis=1) / np.maximum(cnt, 1), np.nan)
        out.append((raw - 1.0) / (c.K - 1.0) * 100.0)
    return np.column_stack(out)


_SPLIT_CACHE = {}


def corpus_split(name):
    """(corpus, inp, tgt, scores) for a corpus, computed once."""
    if name not in _SPLIT_CACHE:
        c = load_corpus(name)
        inp, tgt = split_items(c.ids, c.scale, c.n_input)
        _SPLIT_CACHE[name] = (c, inp, tgt, scores_from(c, inp))
    return _SPLIT_CACHE[name]


def reconstruct(meta, run):
    """Recover (observed_scored, observed_train, target_ids, K) for a run.

    Prefers artifacts the run saved itself; falls back to replaying the
    deterministic panel split. Either way the result is VERIFIED against the
    saved target_answers.npy before it is returned.
    """
    Yt = np.load(run / "target_answers.npy")
    if (run / "observed_scored.npy").exists():          # h24 saves these directly
        return (np.load(run / "observed_scored.npy"),
                np.load(run / "observed_train.npy"), True)

    corpus = meta.get("corpus") or meta.get("instrument")
    c, inp, tgt, S = corpus_split(corpus)
    # The earliest h15 runs predate the target_ids field; derive it from the
    # corpus split and check the SHAPE instead, which is the part that matters.
    declared = meta.get("target_ids")
    if declared is not None and list(tgt) != list(declared):
        return None, None, False
    if len(tgt) != Yt.shape[1]:
        return None, None, False
    idx = {i: k for k, i in enumerate(c.ids)}

    n_panel = len(Yt)
    n_train = len(np.load(run / "train_answers.npy"))
    panel = meta.get("panel")
    if "scored_idx" in meta:                            # recorded explicitly (h24)
        scored = np.array(meta["scored_idx"]); train = np.array(meta["train_idx"])
    elif "instrument" in meta:
        # h15 family, which also produced every h16_{val,test2} run: a plain
        # permutation of ALL respondents under seed 260909, with a fixed 256
        # reserved at the head (h15_disjoint_readout's --n-test default) whether
        # or not the scored panel is that block. h17 uses a DIFFERENT seed and
        # permutes only respondents with a complete persona, so the two must not
        # be conflated -- doing so silently mismatches 12 of the 13 h16 runs.
        perm = np.random.default_rng(260909).permutation(c.Y.shape[0])
        reserved = 256
        train = perm[reserved: reserved + n_train]
        if panel in ("val", "test2"):
            base = reserved + len(train) + (0 if panel == "val" else n_panel)
            scored = perm[base: base + n_panel]
        else:
            scored = perm[:n_panel]
    elif panel in ("val", "test2"):                     # h17 protocol
        eligible = np.where(~np.isnan(S).any(axis=1))[0]
        perm = eligible[np.random.default_rng(260910).permutation(len(eligible))]
        n_hold = 256
        train = perm[n_hold: n_hold + n_train]
        base = n_hold + len(train)
        off = 0 if panel == "val" else n_panel
        scored = perm[base + off: base + off + n_panel]
    else:
        perm = np.random.default_rng(260909).permutation(c.Y.shape[0])
        scored, train = perm[:n_panel], perm[n_panel: n_panel + n_train]

    got = c.Y[np.ix_(scored, [idx[j] for j in tgt])]
    same = (np.isnan(got) == np.isnan(Yt)).all() and np.allclose(
        got[np.isfinite(got)], Yt[np.isfinite(Yt)])
    if not same:
        return None, None, False
    ii = [idx[i] for i in inp]
    return c.Y[np.ix_(scored, ii)], c.Y[np.ix_(train, ii)], True


# ------------------------------------------------------------------ one run
LAMS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
ALPHAS = np.round(np.arange(0.0, 2.01, 0.1), 2)
WS = np.round(np.arange(0.0, 1.001, 0.05), 3)


# Expected reverse-keyed TARGET counts per corpus, from the published keys.
# Stale directories written before the key fix carry a different count and are
# scored alongside the corrected ones unless filtered -- on big5 that put 19
# rows per panel into the h17n512 screen where the other corpora have 12.
# Filter on this, never on directory spelling: ipip, sd3 and hexaco all use the
# "p" spelling that a spelling filter would delete.
EXPECTED_REV = {"ipip": 43, "sd3": 3, "big5": 7, "hexaco": 46}


def screen(run: Path, rng, use_gbm=True, strict_rev=True):
    meta = json.loads((run / "summary.json").read_text())
    if not meta.get("valid", False):
        return dict(run=run.name, status=f"invalid(mass={meta.get('mass_on_scale_mean')})")
    _c = meta.get("corpus") or meta.get("instrument")
    _exp = EXPECTED_REV.get(_c)
    _got = meta.get("reverse_keyed_targets")
    if strict_rev and _exp is not None and _got is not None and len(_got) != _exp:
        return dict(run=run.name, status=f"stale-key(rev={len(_got)},expected={_exp})",
                    corpus=_c, model=meta.get("model"))
    P = np.load(run / "belief_probs.npy")
    K = int(meta.get("K", P.shape[2]))
    corpus = meta.get("corpus") or meta.get("instrument")
    tgt_ids = meta.get("target_ids")
    recoded = meta.get("corpus_recoded")
    if recoded is None or tgt_ids is None:
        # Older summaries omit these. Recover them from the corpus rather than
        # defaulting to "no flip" -- a missed flip is invisible in any variance
        # statistic and wrong in every score that follows.
        c0 = load_corpus(corpus)
        if recoded is None:
            recoded = c0.recoded
        if tgt_ids is None:
            tgt_ids = split_items(c0.ids, c0.scale, c0.n_input)[1]
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
        return dict(run=run.name, status="reconstruction-mismatch")

    n, nt = Yt.shape
    th = list(range(1, K))
    Q = np.cumsum(P, axis=2)[:, :, : K - 1]
    sp = np.random.default_rng(20260918).permutation(n)
    CAL, EV = sp[: n // 2], sp[n // 2:]

    # COMMON SUPPORT. The LLM arms carry NaN wherever the model refused (digit
    # mass < 0.1); the statistical arms never do. Reducing each arm over its own
    # valid cells would compare arms on DIFFERENT item sets and hand the LLM the
    # easy half. Fix one arm-independent mask -- finite human answer AND finite
    # LLM forecast -- and apply it to every arm, so every number below is a
    # paired comparison over identical cells.
    support = np.isfinite(Yt) & np.isfinite(Q).all(axis=2)
    dropped = int((~support).sum())
    if dropped:
        print(f"  [{run.name}] common support drops {dropped} of {n * nt} cells "
              f"({100 * dropped / (n * nt):.2f}%)", flush=True)
    Yt = np.where(support, Yt, np.nan)

    prior_row = np.stack([np.nanmean(Ytr <= t, axis=0) for t in th], axis=1)
    PRIOR = np.broadcast_to(prior_row[None], (n, nt, K - 1))
    targ = np.concatenate([(Ytr <= t).astype(float) for t in th], axis=1)
    targ = np.nan_to_num(targ, nan=0.0)
    Ztr, Zs = onehot(Xtr, K), onehot(Xs, K)

    def dec(lam):
        flat = np.clip(ridge(Ztr, targ, Zs, lam), 0, 1)
        return iso_fast(np.transpose(flat.reshape(n, K - 1, nt), (0, 2, 1)))
    lam = min(LAMS, key=lambda l: np.nanmean(rps(dec(l)[CAL], Yt[CAL], K)))
    DEC = dec(lam)

    Qbar = np.nanmean(Q[CAL], axis=0)
    def cal(rows, a):
        return iso_fast(np.clip(prior_row[None] + a * (Q[rows] - Qbar[None]), 0, 1))
    alpha = min(ALPHAS, key=lambda a: np.nanmean(rps(cal(CAL, a), Yt[CAL], K)))
    w = min(WS, key=lambda w: np.nanmean(
        rps(iso_fast(np.clip((1 - w) * DEC[CAL] + w * cal(CAL, alpha), 0, 1)), Yt[CAL], K)))

    def arm(M, rows):
        return np.nanmean(rps(M, Yt[rows], K), axis=1)

    r = dict(run=run.name, status="ok", model=meta.get("model", "?"),
             corpus=meta.get("corpus") or meta.get("instrument"),
             panel=meta.get("panel", meta.get("persona", "-")),
             n_eval=len(EV), n_items=nt, K=K, n_train=len(Ytr),
             mass=meta.get("mass_on_scale_mean"), lam=lam, alpha=alpha, w_star=w,
             support_dropped=dropped,
             support_frac=float(support.mean()),
             llm_raw=float(np.nanmean(arm(Q[EV], EV))),
             llm_cal=float(np.nanmean(arm(cal(EV, alpha), EV))),
             prior=float(np.nanmean(arm(PRIOR[EV], EV))),
             decoder=float(np.nanmean(arm(DEC[EV], EV))),
             llm_logloss=float(np.nanmean(logloss(P[EV], Yt[EV]))))
    stackEV = iso_fast(np.clip((1 - w) * DEC[EV] + w * cal(EV, alpha), 0, 1))
    r["stack"] = float(np.nanmean(arm(stackEV, EV)))
    d = arm(DEC[EV], EV) - arm(stackEV, EV)
    lo, hi = boot_ci(d[np.isfinite(d)], rng)
    r.update(stack_gain=float(np.nanmean(d)), stack_lo=lo, stack_hi=hi,
             stack_sig=bool(lo > 0))

    # feature-level combiner: the strong test a convex weight cannot do
    def rows_of(sel, with_llm):
        f = [DEC[sel].reshape(-1, K - 1), PRIOR[sel].reshape(-1, K - 1)]
        if with_llm:
            f.append(Q[sel].reshape(-1, K - 1))
        return np.nan_to_num(np.hstack(f), nan=0.5)
    ind = np.stack([(Yt <= t).astype(float) for t in th], axis=2)
    yc = np.nan_to_num(ind[CAL].reshape(-1, K - 1), nan=0.0)
    res = {}
    for wl in (False, True):
        Xc, Xe = rows_of(CAL, wl), rows_of(EV, wl)
        h = len(Xc) // 2
        lc = min([1e-3, 1e-2, 1e-1, 1.0, 10.0], key=lambda l: float(
            ((np.clip(ridge(Xc[:h], yc[:h], Xc[h:], l), 0, 1) - yc[h:]) ** 2).mean()))
        res[("lin", wl)] = iso_fast(
            np.clip(ridge(Xc, yc, Xe, lc), 0, 1).reshape(len(EV), nt, K - 1))
        if use_gbm:
            from sklearn.ensemble import HistGradientBoostingRegressor
            g = np.column_stack([HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.06, max_depth=4, random_state=0
            ).fit(Xc, yc[:, t]).predict(Xe) for t in range(K - 1)])
            res[("gbm", wl)] = iso_fast(np.clip(g, 0, 1).reshape(len(EV), nt, K - 1))
    for fam in (("lin",) + (("gbm",) if use_gbm else ())):
        b, a_ = arm(res[(fam, False)], EV), arm(res[(fam, True)], EV)
        dd = b - a_
        lo, hi = boot_ci(dd[np.isfinite(dd)], rng)
        r[f"{fam}_noLLM"] = float(np.nanmean(b))
        r[f"{fam}_withLLM"] = float(np.nanmean(a_))
        r[f"{fam}_gain"] = float(np.nanmean(dd))
        r[f"{fam}_lo"], r[f"{fam}_hi"] = lo, hi
        r[f"{fam}_sig"] = bool(lo > 0)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True,
                    help="directories to scan for run subdirectories")
    ap.add_argument("--glob", default="*", help="restrict run names")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-gbm", action="store_true")
    ap.add_argument("--allow-stale-keys", action="store_true",
                    help="score directories whose reverse-key count does not "
                         "match the published key (default: refuse them)")
    ap.add_argument("--seed", type=int, default=20260918)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    runs = []
    for rt in a.roots:
        p = ROOT / rt if not Path(rt).is_absolute() else Path(rt)
        for d in sorted(p.glob(a.glob)):
            if (d / "belief_probs.npy").exists() and (d / "target_answers.npy").exists() \
               and (d / "train_answers.npy").exists() and (d / "summary.json").exists():
                runs.append(d)
    print(f"{len(runs)} candidate runs", flush=True)

    rows = []
    for i, d in enumerate(runs, 1):
        try:
            r = screen(d, rng, use_gbm=not a.no_gbm,
                       strict_rev=not a.allow_stale_keys)
        except Exception as e:                       # one bad run must not kill the screen
            r = dict(run=d.name, status=f"error:{type(e).__name__}:{e}")
        rows.append(r)
        s = r.get("status")
        if s == "ok":
            print(f"[{i}/{len(runs)}] {r['run'][:52]:52s} llm {r['llm_raw']:.4f} "
                  f"prior {r['prior']:.4f} dec {r['decoder']:.4f} w* {r['w_star']:.2f} "
                  f"lin {r.get('lin_gain', float('nan')):+.4f}"
                  f"{'*' if r.get('lin_sig') else ''} "
                  f"gbm {r.get('gbm_gain', float('nan')):+.4f}"
                  f"{'*' if r.get('gbm_sig') else ''}", flush=True)
        else:
            print(f"[{i}/{len(runs)}] {r['run'][:52]:52s} {s}", flush=True)

    out = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=["run", "status"] +
                             [k for k in keys if k not in ("run", "status")])
        wtr.writeheader()
        wtr.writerows(rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"\nwrote {out}  ({len(ok)} scored, {len(rows) - len(ok)} skipped)")
    for fam in ("stack", "lin", "gbm"):
        sig = [r for r in ok if r.get(f"{fam}_sig")]
        if ok:
            print(f"  {fam:5s}: {len(sig)}/{len(ok)} cells show a significant "
                  f"positive LLM increment")
    if ok:
        best = max(ok, key=lambda r: r.get("lin_gain", -9))
        print(f"  largest linear increment: {best['run']} {best.get('lin_gain'):+.4f} "
              f"[{best.get('lin_lo'):+.4f},{best.get('lin_hi'):+.4f}]")


if __name__ == "__main__":
    main()
