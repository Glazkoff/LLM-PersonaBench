"""H20: is a belief readout reproducible across independent stacks?

Every H17 number assumes a readout is a stable measurement of a model. This
paper's own Limitations report the readout PATH moving answer-token mass from
0.39 to 0.99, so that assumption is not free. Same script, seed and respondent
panels; different cluster, GPU, torch and transformers (5.17.0 on HSE vs
5.16.1 on Euler).

Reports max and mean absolute deviation over the full belief simplex, whether
the two runs would rank identically under S_0 and S_1/2, and whether any
respondent's modal answer flips.
"""
import glob, json, os
import numpy as np

# Paths are overridable so the released bundle can be pointed at from any
# checkout: set ARR_HSE_REPL / ARR_EULER_RESULTS to the unpacked directories.
HSE = os.environ.get("ARR_HSE_REPL",
                     "/home/glazkov/personality-twins-arr/hse_repl")
EUL = os.environ.get("ARR_EULER_RESULTS",
                     "/home/glazkov/personality-twins-arr/LLM-PersonaBench/arr2026/results_euler")


def index_by_identity(root, prefix):
    """Key cells by (model, corpus, panel) from summary.json, not by directory
    name. The two clusters tag differently -- Euler writes Apertus-v1p1-0p5B
    (dots -> p), HSE writes Apertus-v1_1-0_5B (dots -> _) -- and matching on
    munged names silently dropped half the comparison as NO EULER TWIN."""
    out = {}
    for d in glob.glob(os.path.join(root, prefix + "*")):
        f = os.path.join(d, "summary.json")
        if not os.path.exists(f):
            continue
        try:
            m = json.load(open(f))
        except Exception:
            continue
        out[(m.get("model"), m.get("corpus"), m.get("panel"))] = d
    return out


print(f"{'cell':46s} {'maxdev':>10s} {'meandev':>10s} {'mode flips':>11s}")
rows = []
hse_idx = index_by_identity(HSE, "h17hse_")
eul_idx = index_by_identity(EUL, "h17n512_")
for key in sorted(hse_idx, key=lambda k: (str(k[0]), str(k[1]))):
    d = hse_idx[key]
    name = f"{str(key[0]).split('/')[-1]} / {key[1]}"
    tw = eul_idx.get(key)
    if tw is None:
        print(f"{name:46s} NO EULER TWIN for {key}"); continue
    a = np.load(os.path.join(d, "belief_probs.npy"))
    b = np.load(os.path.join(tw, "belief_probs.npy"))
    if a.shape != b.shape:
        print(f"{name:46s} SHAPE {a.shape} vs {b.shape}"); continue
    dev = np.abs(a - b)
    flips = int((np.nanargmax(a, axis=2) != np.nanargmax(b, axis=2)).sum())
    print(f"{name:46s} {np.nanmax(dev):10.2e} {np.nanmean(dev):10.2e} "
          f"{flips:11d}")
    rows.append((name, float(np.nanmax(dev)), float(np.nanmean(dev)), flips,
                 int(a.shape[0] * a.shape[1])))

if rows:
    worst = max(r[1] for r in rows)
    tot_flip = sum(r[3] for r in rows); tot_cell = sum(r[4] for r in rows)
    print(f"\nworst max deviation across cells: {worst:.2e}")
    print(f"modal-answer flips: {tot_flip} of {tot_cell} item-respondent pairs "
          f"({100.0*tot_flip/max(tot_cell,1):.4f}%)")
    print("bit-identical" if worst == 0.0 else
          ("numerically identical to float tolerance" if worst < 1e-6 else
           "DIVERGENT -- the readout is not stack-invariant"))
