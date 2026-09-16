"""
H23 -- how stable is the H21 REACHABLE/NOT SEPARATED verdict?

H21 asks whether prompt search can reach a candidate that S_0 prefers and a
proper rule rejects. Its verdict is a strict inequality with a 1e-6 threshold:

    REACHABLE iff sd[argmax_S0] < sd[argmin_CRPS] - 1e-6

On the first full run two of four models fired REACHABLE on CRPS gaps of 0.0002
and 0.0009 between the same two top conditions, which merely swapped order. A
strict inequality with no uncertainty cannot tell a real separation from a tie,
and H22 showed concretely that a gap of that scale in a related quantity can
straddle zero. This script supplies the missing uncertainty.

Resampling unit: the RESPONDENT. All twelve conditions are evaluated on the same
respondents, so conditions are paired by construction; each replicate draws one
set of respondent indices and applies it to every condition. The selection is
re-run inside each replicate (argmax/argmin recomputed, not frozen from the
point estimate) so the reported stability includes selection uncertainty, which
is the thing actually in doubt.

Reads per_respondent.json, written by h21_reachability.py.
"""
import argparse, json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def verdict(s0m, crpsm, sdm, keys):
    p_s0 = keys[int(np.argmax(s0m))]
    p_cr = keys[int(np.argmin(crpsm))]
    i, j = keys.index(p_s0), keys.index(p_cr)
    return p_s0, p_cr, bool(sdm[i] < sdm[j] - 1e-6), float(sdm[i] - sdm[j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="arr2026/results_euler")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="arr2026/results_euler/h23_reachability_ci")
    a = ap.parse_args()

    out = {"n_boot": a.n_boot, "seed": a.seed, "models": {}}
    for tag in a.tags:
        p = ROOT / a.results / tag / "per_respondent.json"
        if not p.exists():
            print(f"MISSING {p} -- rerun h21 to emit per-respondent scores")
            continue
        d = json.loads(p.read_text())
        pr = d["per_respondent"]
        keys = list(pr.keys())
        S0 = np.array([pr[k]["s0"] for k in keys], float)      # (cond, resp)
        CR = np.array([pr[k]["crps"] for k in keys], float)
        SD = np.array([pr[k]["sd"] for k in keys], float)
        n = S0.shape[1]
        print(f"\n===== {d['model']} ({len(keys)} conditions, {n} respondents) =====",
              flush=True)

        p_s0, p_cr, reach, gap = verdict(np.nanmean(S0, 1), np.nanmean(CR, 1),
                                         np.nanmean(SD, 1), keys)
        print(f"point estimate: S_0 -> {p_s0} | CRPS -> {p_cr}")
        print(f"  verdict={'REACHABLE' if reach else 'NOT SEPARATED'} "
              f"sd gap={gap:+.4f}")

        rng = np.random.default_rng(a.seed)
        n_reach = 0
        gaps, agree = [], 0
        picks_s0, picks_cr = {}, {}
        for _ in range(a.n_boot):
            idx = rng.integers(0, n, n)
            q_s0, q_cr, r, g = verdict(np.nanmean(S0[:, idx], 1),
                                       np.nanmean(CR[:, idx], 1),
                                       np.nanmean(SD[:, idx], 1), keys)
            n_reach += r
            gaps.append(g)
            agree += (q_s0 == q_cr)
            picks_s0[q_s0] = picks_s0.get(q_s0, 0) + 1
            picks_cr[q_cr] = picks_cr.get(q_cr, 0) + 1

        frac = n_reach / a.n_boot
        lo, hi = np.percentile(gaps, [2.5, 97.5])
        print(f"  REACHABLE fires in {frac:.1%} of {a.n_boot} resamples")
        print(f"  sd(S_0 pick) - sd(CRPS pick): {gap:+.4f} [{lo:+.4f}, {hi:+.4f}]"
              f"  {'excludes 0' if (lo > 0 or hi < 0) else 'INCLUDES 0'}")
        print(f"  the two rules agree on the same condition in {agree/a.n_boot:.1%}")
        # When the argmax/argmin themselves flip between replicates the sd gap is
        # a mixture of two sign-flipped modes, so its percentile interval is not
        # an "excludes zero" test of anything. Say so rather than let the bound
        # be read as one. REACHABLE-fraction and the pick distributions stay
        # interpretable in that regime; the gap CI does not.
        stab = max(max(picks_s0.values()), 0) / a.n_boot
        stab_cr = max(max(picks_cr.values()), 0) / a.n_boot
        if min(stab, stab_cr) < 0.90:
            print(f"  NOTE: selection is unstable (top pick held "
                  f"{min(stab, stab_cr):.0%} of replicates); the sd-gap interval "
                  f"above is bimodal and must NOT be read as a significance test. "
                  f"Use the REACHABLE fraction and the pick distributions.")
        top_s0 = sorted(picks_s0.items(), key=lambda x: -x[1])[:3]
        top_cr = sorted(picks_cr.items(), key=lambda x: -x[1])[:3]
        print(f"  S_0 selects:  " + ", ".join(f"{k} {v/a.n_boot:.0%}" for k, v in top_s0))
        print(f"  CRPS selects: " + ", ".join(f"{k} {v/a.n_boot:.0%}" for k, v in top_cr))

        out["models"][tag] = {
            "model": d["model"], "n_respondents": n,
            "point_pick_s0": p_s0, "point_pick_crps": p_cr,
            "point_verdict_reachable": reach, "point_sd_gap": gap,
            "reachable_fraction": frac,
            "sd_gap_lo": float(lo), "sd_gap_hi": float(hi),
            "sd_gap_excludes_zero": bool(lo > 0 or hi < 0),
            "rules_agree_fraction": agree / a.n_boot,
            "selection_stability": float(min(stab, stab_cr)),
            "sd_gap_ci_interpretable": bool(min(stab, stab_cr) >= 0.90),
            "s0_pick_distribution": {k: v / a.n_boot for k, v in picks_s0.items()},
            "crps_pick_distribution": {k: v / a.n_boot for k, v in picks_cr.items()},
        }

    dd = ROOT / a.out
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dd/'summary.json'}")


if __name__ == "__main__":
    main()
