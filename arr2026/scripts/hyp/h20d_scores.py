"""Does the ARCHITECTURE change the SCORES, or only the argmax?

7.7% of modal answers flip between A100 and H200, concentrated on near-ties.
The paper's operational advice is to score the distribution a simulator
induces, not the answer it would most likely give. If that advice is right,
distribution-based scores should be far more stable across hardware than mode
accuracy is -- which is a second, independent argument for it, orthogonal to
propriety.
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


def s0(P, Y):
    K = P.shape[2]; lv = np.arange(1, K + 1, dtype=float)
    d = np.abs(lv[None, None, :] - Y[:, :, None]) / (K - 1)
    return float(np.nanmean(np.where(np.isnan(Y), np.nan, np.nansum(P * (1 - d), 2))))


def crps(P, Y):
    K = P.shape[2]
    Q = np.cumsum(P, 2)[:, :, :K - 1]
    ind = np.stack([(Y <= t).astype(float) for t in range(1, K)], axis=2)
    return float(np.nanmean(np.where(np.isnan(Y), np.nan,
                                     np.nansum((Q - ind) ** 2, 2) / (K - 1))))


def mode_acc(P, Y):
    pred = np.nanargmax(P, axis=2) + 1
    return float(np.nanmean(np.where(np.isnan(Y), np.nan, (pred == Y).astype(float))))


a, e = idx(A100, "h17a100_"), idx(EUL, "h17n512_")
print(f"{'cell':44s} {'metric':11s} {'H200':>9s} {'A100':>9s} {'|diff|':>9s}")
agg = {}
for k in sorted(a, key=lambda x: str(x)):
    if k not in e:
        continue
    Pa = np.load(os.path.join(a[k], "belief_probs.npy"))
    Pe = np.load(os.path.join(e[k], "belief_probs.npy"))
    Y = np.load(os.path.join(e[k], "target_answers.npy"))
    name = f"{str(k[0]).split('/')[-1]} / {k[1]}"
    for lab, fn in (("S_1/2", crps), ("S_0", s0), ("mode acc", mode_acc)):
        va, ve = fn(Pa, Y), fn(Pe, Y)
        print(f"{name:44s} {lab:11s} {ve:9.5f} {va:9.5f} {abs(va-ve):9.2e}")
        agg.setdefault(lab, []).append(abs(va - ve))
    print()
print("mean |A100 - H200| by metric:")
for lab, v in agg.items():
    print(f"  {lab:11s} {np.mean(v):.2e}")
