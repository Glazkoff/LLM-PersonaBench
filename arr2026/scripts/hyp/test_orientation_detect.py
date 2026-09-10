r"""Validate the orientation detector on corpora whose answer is already known.

IPIP-NEO-120 is stored RECODED (the paper's h15 fix established this the hard
way); SD3 is stored RAW, with reverse-vs-forward correlations of -0.19 to -0.31.
A detector that cannot recover those two has no business being pointed at
HEXACO or BIG5.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _orientation_detect import detect_recoded, item_scale_corr  # noqa: E402


def load_ipip():
    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")
    key = pd.read_csv(ROOT / "data/IPIP-NEO/120/item_key.csv")
    col = {c.lower(): c for c in key.columns}
    idc = col.get("id") or col.get("item") or key.columns[0]
    fac = col.get("facet") or col.get("label_raw") or key.columns[1]
    items = [c for c in df.columns if c.lower().startswith("i") and c[1:].isdigit()]
    items = sorted(items, key=lambda c: int(c[1:]))[:120]
    scale = list(key.sort_values(idc)[fac].astype(str))[:120]
    return df[items].to_numpy(float), scale


def load_sd3():
    df = pd.read_csv(ROOT / "data/PAlign/Dark-Triad.csv", sep="\t")
    items = [c for c in df.columns if len(c) <= 3 and c[0] in "MNP" and c[1:].isdigit()]
    Y = df[items].to_numpy(float)
    Y[(Y < 1) | (Y > 5)] = np.nan
    return Y, [c[0] for c in items]


def main() -> None:
    bad = []
    for name, loader, expect in (("IPIP-NEO-120", load_ipip, True),
                                 ("SD3", load_sd3, False)):
        try:
            Y, scale = loader()
        except Exception as e:
            print(f"{name:14s} LOAD FAILED: {e}"); bad.append(f"{name} load"); continue
        rec, frac_neg, mean_r = detect_recoded(Y, scale)
        r = item_scale_corr(Y, scale); v = r[~np.isnan(r)]
        print(f"{name:14s} n={Y.shape[0]:>7d} items={Y.shape[1]:>3d} "
              f"scales={len(set(scale)):>3d} | neg={frac_neg:.3f} mean_r={mean_r:+.3f} "
              f"range=[{v.min():+.2f},{v.max():+.2f}] -> recoded={rec} (expect {expect})")
        if rec != expect:
            bad.append(f"{name}: detector said recoded={rec}, known {expect}")
    if bad:
        print("\nDETECTOR INVALID:"); [print("  -", b) for b in bad]
        raise SystemExit(1)
    print("\nDetector reproduces both known corpora; safe to apply to a new one.")


if __name__ == "__main__":
    main()
