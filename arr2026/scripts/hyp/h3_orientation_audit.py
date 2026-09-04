"""
H3 -- Orientation audit: are model and human answers on the same scale?

The human answer columns in df_ipipneo_120_clusters appear to be stored
REVERSE-RECODED (higher = more of the trait), while the simulator is shown the
literal IPIP item text ("Cheat to get ahead") and asked how accurately it
describes it -- i.e. it answers RAW. On the 55 negatively-keyed items of the
IPIP-NEO-120 the two scales then run in opposite directions, and every
model-vs-human comparison is computed across that flip.

Evidence for the recoding:
  * negatively-keyed items average 3.373 vs 3.425 for positively-keyed ones;
    in raw self-reports undesirable items sit far lower, not level.
  * "Cheat to get ahead" has a human mean of 4.15/5.

This script recomputes the head-to-head under both orientations and reports
which of our findings are orientation-robust. Predictions, stated before
running:
  * VR is EXACTLY invariant, since sd(6-x) = sd(x).
  * The Proposition (MAD, GMD) is invariant -- both are computed on the human
    distribution alone and both are translation/reflection invariant.
  * S_answer and C2ST are NOT invariant, and the model is the party penalised,
    because the constant baseline is built from the human data and therefore
    already lives in the human orientation.

CPU only. No LLM calls.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ITEMS = [f"i{i}" for i in range(1, 121)]
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "arr2026/results/h3"


def neg_mask():
    k = pd.read_excel(ROOT / "data/PAlign/IPIP-NEO-ItemKey.xls")
    k = k[k["Short#"].notna()].copy()
    k["Short#"] = k["Short#"].astype(int)
    k = k[k["Short#"] <= 120]
    m = np.zeros(120, dtype=bool)
    for _, r in k.iterrows():
        if str(r["Sign"]).strip().startswith("-"):
            m[int(r["Short#"]) - 1] = True
    return m


NEG = neg_mask()


def sim(pred, human):
    return float(np.nanmean(1.0 - np.abs(pred - human) / 4.0))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")
    by_case = df.set_index("case")
    print(f"negatively-keyed items: {NEG.sum()}/120")

    rows = []
    for cdir in sorted((ROOT / "results_experiments/evoprompt_iter2").glob("*/*/cluster_*")):
        cid = int(cdir.name.split("_")[1]); model = cdir.parent.parent.name
        f = cdir / "after_optimization_test_answers.csv"
        tr_f, te_f = cdir / "train_case_ids.csv", cdir / "test_case_ids.csv"
        if not (f.exists() and te_f.exists() and tr_f.exists()):
            continue
        M = pd.read_csv(f)[ITEMS].to_numpy(float)
        if np.isnan(M).mean() > 0.5 or len(np.unique(M[~np.isnan(M)])) < 2:
            continue
        te = pd.read_csv(te_f)["case"].tolist(); tr = pd.read_csv(tr_f)["case"].tolist()
        H = by_case.loc[[c for c in te if c in by_case.index]][ITEMS].to_numpy(float)
        T = by_case.loc[[c for c in tr if c in by_case.index]][ITEMS].to_numpy(float)
        if len(H) == 0 or len(T) == 0:
            continue

        # model mapped into the human (recoded) orientation
        Mc = M.copy(); Mc[:, NEG] = 6.0 - Mc[:, NEG]
        const = np.tile(np.round(T.mean(axis=0)), (len(H), 1))
        n = min(len(M), len(H))

        row = {
            "model": model, "cluster": cid,
            "S_model_asis": sim(M[:n], H[:n]),
            "S_model_reoriented": sim(Mc[:n], H[:n]),
            "S_constant": sim(const, H),
            "VR_asis": float(np.nanmean(np.nanstd(M, 0)[np.nanstd(H, 0) > 0]
                                        / np.nanstd(H, 0)[np.nanstd(H, 0) > 0])),
            "VR_reoriented": float(np.nanmean(np.nanstd(Mc, 0)[np.nanstd(H, 0) > 0]
                                              / np.nanstd(H, 0)[np.nanstd(H, 0) > 0])),
            # does the model's per-item mean profile track the humans'?
            "corr_itemmeans_asis": float(np.corrcoef(np.nanmean(M, 0), np.nanmean(H, 0))[0, 1]),
            "corr_itemmeans_reoriented": float(np.corrcoef(np.nanmean(Mc, 0), np.nanmean(H, 0))[0, 1]),
        }
        rows.append(row)
        print(f"  {model:13s} c{cid} S: {row['S_model_asis']:.4f} -> "
              f"{row['S_model_reoriented']:.4f} (const {row['S_constant']:.4f}) | "
              f"corr {row['corr_itemmeans_asis']:+.3f} -> "
              f"{row['corr_itemmeans_reoriented']:+.3f}", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(OUT / "orientation.csv", index=False)
    summary = {
        "n_cells": int(len(t)),
        "n_negatively_keyed_items": int(NEG.sum()),
        "S_model_asis": round(float(t.S_model_asis.mean()), 4),
        "S_model_reoriented": round(float(t.S_model_reoriented.mean()), 4),
        "S_constant": round(float(t.S_constant.mean()), 4),
        "cells_model_beats_constant_asis": int((t.S_model_asis > t.S_constant).sum()),
        "cells_model_beats_constant_reoriented": int((t.S_model_reoriented > t.S_constant).sum()),
        "VR_invariant_as_predicted": bool(np.allclose(t.VR_asis, t.VR_reoriented, atol=1e-9)),
        "corr_itemmeans_asis": round(float(t.corr_itemmeans_asis.mean()), 4),
        "corr_itemmeans_reoriented": round(float(t.corr_itemmeans_reoriented.mean()), 4),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== H3 ORIENTATION SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
