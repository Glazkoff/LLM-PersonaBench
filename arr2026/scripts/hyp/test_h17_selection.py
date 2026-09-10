r"""Does the selection machinery recover an answer we planted?

Build three synthetic candidates on a known truth:
  calibrated  forecasts the true per-item distribution
  sharpened   the truth raised to a power and renormalised -- overconfident
  degenerate  a point mass at the truth's median, the Prop. 1 optimum for S_0

If the pipeline is wired correctly then, on held-out respondents, the strict
rules must select `calibrated` and s0 must select `degenerate`. If s0 picks the
calibrated model, the harness is not reproducing the defect the paper is about
and any real result from it would be uninterpretable.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _scoring import RULES, STRICT, score  # noqa: E402


def build(K=5, n=400, J=30, seed=7):
    rng = np.random.default_rng(seed)
    truth = rng.dirichlet(np.full(K, 0.8), size=J)            # per-item truth
    Y = np.stack([rng.choice(np.arange(1, K + 1), size=n, p=truth[j])
                  for j in range(J)], axis=1).astype(float)
    cal = np.broadcast_to(truth, (n, J, K)).copy()
    sharp = truth ** 3
    sharp = np.broadcast_to(sharp / sharp.sum(1, keepdims=True), (n, J, K)).copy()
    med = np.argmax(np.cumsum(truth, axis=1) >= 0.5, axis=1)
    deg = np.zeros((J, K)); deg[np.arange(J), med] = 1.0
    deg = np.broadcast_to(deg, (n, J, K)).copy()
    return Y, {"calibrated": cal, "sharpened": sharp, "degenerate": deg}


def main() -> None:
    bad = []
    for K in (5, 7):
        Y, cands = build(K=K)
        means = {m: {r: float(np.nanmean(score(r, P, Y))) for r in RULES}
                 for m, P in cands.items()}
        print(f"\n--- K={K} ---")
        print(f"{'selector':12s} {'picks':12s}   " +
              "  ".join(f"{m[:9]:>9s}" for m in cands))
        for r in RULES:
            pick = min(cands, key=lambda m: means[m][r])
            vals = "  ".join(f"{means[m][r]:>9.4f}" for m in cands)
            print(f"{r:12s} {pick:12s}   {vals}")
            if r in STRICT and pick != "calibrated":
                bad.append(f"K={K}: strict rule {r} picked {pick}, not calibrated")
            if r == "s0" and pick != "degenerate":
                bad.append(f"K={K}: s0 picked {pick}; the S_0 defect is not "
                           f"being reproduced, so the harness proves nothing")
    if bad:
        print("\nHARNESS INVALID:"); [print("  -", b) for b in bad]
        raise SystemExit(1)
    print("\nEvery strict rule selects the calibrated candidate; s0 selects the "
          "degenerate one at both scale lengths. Harness recovers the planted answer.")


if __name__ == "__main__":
    main()
