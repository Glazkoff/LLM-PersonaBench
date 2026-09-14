"""Is the A100/H200 divergence real signal or a mismatched comparison?

A 7.7% modal-flip rate with a mean deviation of only ~1e-2 is the signature of
near-ties: the model is nearly indifferent between adjacent levels and a tiny
numeric difference tips the argmax. But it is ALSO what you would see if the
two runs scored different respondents. Check that first.
"""
import glob, json, os
import numpy as np

A100 = "/home/glazkov/personality-twins-arr/a100_repl"
EUL = "/home/glazkov/personality-twins-arr/LLM-PersonaBench/arr2026/results_euler"


def idx(root, prefix):
    out = {}
    for d in glob.glob(os.path.join(root, prefix + "*")):
        f = os.path.join(d, "summary.json")
        if os.path.exists(f):
            m = json.load(open(f))
            out[(m.get("model"), m.get("corpus"), m.get("panel"))] = d
    return out


a, e = idx(A100, "h17a100_"), idx(EUL, "h17n512_")
for k in sorted(a, key=lambda x: str(x)):
    if k not in e:
        continue
    da, de = a[k], e[k]
    Ya, Ye = (np.load(os.path.join(x, "target_answers.npy")) for x in (da, de))
    Pa, Pe = (np.load(os.path.join(x, "belief_probs.npy")) for x in (da, de))
    same_people = bool(np.array_equal(np.nan_to_num(Ya, nan=-1), np.nan_to_num(Ye, nan=-1)))
    d = np.abs(Pa - Pe)
    flip = np.nanargmax(Pa, axis=2) != np.nanargmax(Pe, axis=2)
    # among flipped cells, how close were the top two levels on H200?
    srt = np.sort(Pe, axis=2)
    gap = srt[:, :, -1] - srt[:, :, -2]
    name = f"{str(k[0]).split('/')[-1]} / {k[1]}"
    print(f"{name:44s} same respondents={same_people}")
    print(f"   flips {int(flip.sum()):5d}  median top-2 gap at flips "
          f"{np.median(gap[flip]) if flip.any() else float('nan'):.4f}  "
          f"vs {np.median(gap[~flip]):.4f} elsewhere")
    print(f"   deviation: mean {d.mean():.2e}  p99 {np.percentile(d, 99):.2e}  "
          f"max {d.max():.2e}")
