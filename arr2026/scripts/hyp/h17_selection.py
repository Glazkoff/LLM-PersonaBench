r"""Which scoring rule, used as a selection criterion, ships the best simulator?

H16 compared two rules on one instrument. This asks the same question of nine
rules on four instruments, and answers it with a selector x evaluator matrix
rather than a single arbiter -- because picking one arbiter would privilege
whichever selector happens to share its criterion.

  1. Each rule selects a model on the VALIDATION panel (argmin; every rule in
     _scoring is oriented lower-is-better so a selector is never inverted).
  2. On the disjoint TEST panel, every model is scored under every rule.
  3. For evaluator e, a model's regret is normalised against what was
     achievable in this candidate set:
         regret_e(m) = (L_e(m) - min_m' L_e) / (max_m' L_e - min_m' L_e)
     0 = e's own favourite, 1 = e's least favourite. Scale-free, so rules with
     different units (nats vs bounded scores) can be averaged.
  4. A selector is graded only by STRICT rules, and never by itself -- grading
     rps by rps is not evidence.

The self-exclusion makes the evaluator count uneven (3 for a strict selector,
4 for the others), so both the self-excluded and the all-evaluator summaries
are printed, with the count, instead of one number that hides it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _scoring import RULES, STRICT, WEAK, IMPROPER, score  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
CLASS = {**{r: "strict" for r in STRICT}, **{r: "weak" for r in WEAK},
         **{r: "improper" for r in IMPROPER}}


def load_run(d: Path):
    meta = json.loads((d / "summary.json").read_text())
    P = np.load(d / "belief_probs.npy")
    if meta.get("corpus_recoded"):
        rk = set(meta["reverse_keyed_targets"])
        flip = np.array([j in rk for j in meta["target_ids"]], dtype=bool)
        if flip.any():
            P[:, flip, :] = P[:, flip, ::-1]
    return meta, P, np.load(d / "target_answers.npy")


def rule_means(runs):
    """{model: {rule: mean per-respondent score}} for one corpus and panel."""
    out = {}
    for name, (meta, P, Y) in runs.items():
        out[name] = {r: float(np.nanmean(score(r, P, Y))) for r in RULES}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="arr2026/results_hse")
    ap.add_argument("--corpora", nargs="+", default=["ipip", "sd3", "big5", "hexaco"])
    a = ap.parse_args()

    per_corpus, skipped = {}, []
    for corpus in a.corpora:
        runs = {"val": {}, "test2": {}}
        for d in sorted((ROOT / a.results).glob(f"h17_*_{corpus}_*")):
            panel = d.name.rsplit("_", 1)[-1]
            if panel not in runs:
                continue
            try:
                meta, P, Y = load_run(d)
            except FileNotFoundError:
                skipped.append(f"{d.name}: incomplete"); continue
            if not meta.get("valid", False):
                skipped.append(f"{d.name}: invalid (mass {meta.get('mass_on_scale_mean')})")
                continue
            model = meta["model"]
            runs[panel][model] = (meta, P, Y)
        both = set(runs["val"]) & set(runs["test2"])
        if len(both) < 2:
            skipped.append(f"{corpus}: only {len(both)} model(s) on both panels")
            continue
        v = rule_means({m: runs["val"][m] for m in both})
        t = rule_means({m: runs["test2"][m] for m in both})
        per_corpus[corpus] = (v, t, sorted(both))

    if skipped:
        print("SKIPPED:"); [print("  -", s) for s in skipped]; print()
    if not per_corpus:
        raise SystemExit("no corpus has two or more models on both panels")

    agg = {r: [] for r in RULES}
    agg_all = {r: [] for r in RULES}
    for corpus, (v, t, models) in per_corpus.items():
        print(f"===== {corpus}: {len(models)} candidates =====")
        # normalised regret per evaluator on the test panel
        reg = {}
        for e in STRICT:
            vals = np.array([t[m][e] for m in models])
            lo, hi = vals.min(), vals.max()
            rng = hi - lo if hi > lo else 1.0
            reg[e] = {m: float((t[m][e] - lo) / rng) for m in models}
        print(f"{'selector':12s} {'class':9s} {'picks':34s} "
              f"{'regret(self-excl)':>18s} {'n':>3s} {'regret(all)':>12s}")
        rows = []
        for r in RULES:
            pick = min(models, key=lambda m: v[m][r])
            excl = [e for e in STRICT if e != r]
            m_excl = float(np.mean([reg[e][pick] for e in excl]))
            m_all = float(np.mean([reg[e][pick] for e in STRICT]))
            agg[r].append(m_excl); agg_all[r].append(m_all)
            rows.append((m_excl, r, pick, m_all, len(excl)))
        for m_excl, r, pick, m_all, n in sorted(rows):
            print(f"{r:12s} {CLASS[r]:9s} {pick.split('/')[-1][:34]:34s} "
                  f"{m_excl:>18.3f} {n:>3d} {m_all:>12.3f}")
        print()

    print("===== across instruments: mean normalised regret =====")
    print(f"{'selector':12s} {'class':9s} {'mean regret':>12s} {'per-corpus':>34s}")
    for r in sorted(RULES, key=lambda r: float(np.mean(agg[r]))):
        pc = " ".join(f"{x:.2f}" for x in agg[r])
        print(f"{r:12s} {CLASS[r]:9s} {np.mean(agg[r]):>12.3f} {pc:>34s}")
    best = min(RULES, key=lambda r: float(np.mean(agg[r])))
    print(f"\nlowest mean regret: {best} ({CLASS[best]})")
    print("Regret is normalised WITHIN this candidate set, so 0 means 'the best "
          "of these models', never 'good'. All candidates may still lose to a "
          "persona-free prior; that comparison is reported separately.")


if __name__ == "__main__":
    main()
