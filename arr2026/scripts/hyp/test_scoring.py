r"""Does each rule actually have the properness we claim?

A strictly proper rule is minimised in expectation by reporting the truth. The
discriminating case for this paper is the one Prop. 1 is about: reporting the
true distribution F versus reporting a point mass at F's median. S_0 prefers
the point mass -- that IS the defect the paper audits -- so a test that only
checked "proper rules behave well" would not be checking anything. This test
requires the improper rules to fail in that specific direction too.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _scoring import RULES, STRICT, WEAK, IMPROPER, ELICITS, LEVELS  # noqa: E402


def expected(rule, forecast, truth):
    """E_{y~truth}[ score(forecast, y) ], exactly, over all five outcomes."""
    P = np.broadcast_to(forecast, (5, 1, 5)).copy()
    Y = LEVELS.reshape(5, 1)
    return float(np.sum(truth * RULES[rule](P, Y)[:, 0]))


def median_point_mass(F):
    c = np.cumsum(F)
    m = int(np.searchsorted(c, 0.5))
    pm = np.zeros(5); pm[min(m, 4)] = 1.0
    return pm


def functional_twin(rule, F):
    """A forecast that is NOT F but matches the functional `rule` elicits.

    If the rule only pins down that functional, this ties with the truth and
    the rule is weakly, not strictly, proper.
    """
    if ELICITS.get(rule) == "mode":
        pm = np.zeros(5); pm[int(np.argmax(F))] = 1.0
        return pm
    if ELICITS.get(rule) == "mean":
        mu = float(np.sum(F * LEVELS))
        lo = int(np.clip(np.floor(mu), 1, 4)); w = mu - lo
        tw = np.zeros(5); tw[lo - 1] = 1.0 - w; tw[lo] = w
        return tw
    return None


def main() -> None:
    rng = np.random.default_rng(20260910)
    fails, prefers_pm = {r: 0 for r in RULES}, {r: 0 for r in RULES}
    ties = {r: 0 for r in RULES}
    trials = 4000
    for _ in range(trials):
        F = rng.dirichlet(np.full(5, rng.uniform(0.3, 3.0)))
        pm = median_point_mass(F)
        if np.allclose(F, pm):
            continue
        G = rng.dirichlet(np.full(5, 1.0))          # arbitrary rival
        for r in RULES:
            truth_s = expected(r, F, F)
            if expected(r, G, F) < truth_s - 1e-12:
                fails[r] += 1                        # beaten by some rival
            if expected(r, pm, F) < truth_s - 1e-12:
                prefers_pm[r] += 1                   # beaten by the median mass
            tw = functional_twin(r, F)
            if tw is not None and not np.allclose(tw, F) \
               and expected(r, tw, F) <= truth_s + 1e-12:
                ties[r] += 1                         # a non-truth forecast ties

    print(f"{'rule':12s} {'beaten':>10s} {'by med.mass':>12s} {'tied by twin':>13s}  verdict")
    bad = []
    for r in RULES:
        if fails[r] > 0:
            verdict = "IMPROPER"
        elif ties[r] > 0:
            verdict = "WEAK"
        else:
            verdict = "STRICT"
        print(f"{r:12s} {fails[r]:>10d} {prefers_pm[r]:>12d} {ties[r]:>13d}  {verdict}")
        claimed = ("STRICT" if r in STRICT else "WEAK" if r in WEAK else "IMPROPER")
        if verdict != claimed:
            bad.append(f"{r}: claimed {claimed}, measured {verdict}")
    # The audited metric must fail in the specific way Prop. 1 describes.
    if prefers_pm["s0"] == 0:
        bad.append("s0 never preferred the median point mass; the audit's premise "
                   "is not reproduced, so this test is not testing anything")
    if bad:
        print("\nMISMATCH:"); [print("  -", b) for b in bad]
        raise SystemExit(1)
    print(f"\nAll {len(RULES)} rules match their declared class. "
          f"s0 prefers the median point mass in {prefers_pm['s0']}/{trials} draws; "
          f"the {len(WEAK)} weakly proper rules are tied by a non-truth forecast "
          f"matching the single functional they elicit.")


if __name__ == "__main__":
    main()
