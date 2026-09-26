import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.optim.orientation import flip_answers
from src.optim.panels import ITEM_COLS, load_corpus

ROOT = Path(__file__).resolve().parents[2]
IDS = list(range(1, 121))


def orientation_regression(min_corr):
    df = load_corpus().set_index("case")
    before, after = [], []
    for f in glob.glob(str(ROOT / "results_experiments/evoprompt_iter2/*/*/cluster_*/after_optimization_test_answers.csv")):
        m = pd.read_csv(f).dropna()
        if m.empty:
            continue
        X = m[ITEM_COLS].to_numpy(float)
        H = df.loc[m["case"], ITEM_COLS].to_numpy(float)
        hm = np.nanmean(H, 0)
        before.append(np.corrcoef(np.nanmean(X, 0), hm)[0, 1])
        after.append(np.corrcoef(np.nanmean(flip_answers(X, IDS), 0), hm)[0, 1])
    b, a = float(np.mean(before)), float(np.mean(after))
    print(f"runs={len(before)} per-item mean corr before={b:.3f} after={a:.3f}")
    return 0 if (a >= min_corr and a > b) else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("orientation_regression")
    o.add_argument("--min-corr", type=float, default=0.60)
    a = ap.parse_args()
    sys.exit(orientation_regression(a.min_corr))


if __name__ == "__main__":
    main()
