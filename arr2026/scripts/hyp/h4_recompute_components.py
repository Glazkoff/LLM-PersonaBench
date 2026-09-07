"""Recompute the within/between columns of the belief table from saved artifacts.

The table in the paper was assembled by hand, and `summary.json` only stores the
*pooled* variance shares. Pooled shares and per-item mean-of-ratios are not the
same statistic (Jensen), so this script recomputes both and reports which one
reproduces the printed numbers. Any new model row must use the same convention
as the six already in the table.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ITEMS = [f"i{i}" for i in range(1, 121)]


def load_corpus() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")


def components(run_dir: Path, df: pd.DataFrame):
    per_item_w, per_item_b, per_item_t, pooled_w, pooled_b = [], [], [], [], []
    for d in sorted(run_dir.glob("readout_cluster_*")):
        cl = int(d.name.rsplit("_", 1)[1])
        probs = np.load(d / "belief_probs.npy")          # (personas, items, 5)
        human_sd = np.nanstd(df[df.clusters == cl][ITEMS].to_numpy(float), axis=0)
        vals = np.arange(1, 6, dtype=float)

        mu = np.nansum(probs * vals[None, None, :], axis=2)
        ex2 = np.nansum(probs * (vals ** 2)[None, None, :], axis=2)
        var_ij = np.clip(ex2 - mu ** 2, 0, None)
        within_var = np.nanmean(var_ij, axis=0)
        between_var = np.nanvar(mu, axis=0)

        ok = human_sd > 0
        per_item_w.append(np.nanmean(np.sqrt(within_var[ok]) / human_sd[ok]))
        per_item_b.append(np.nanmean(np.sqrt(between_var[ok]) / human_sd[ok]))
        per_item_t.append(np.nanmean(np.sqrt(within_var[ok] + between_var[ok]) / human_sd[ok]))
        pooled_w.append(np.nansum(within_var[ok]))
        pooled_b.append(np.nansum(between_var[ok]))
    tot = np.mean(per_item_t)
    sw = np.sum(pooled_w) / (np.sum(pooled_w) + np.sum(pooled_b))
    return {
        "per_item_within": float(np.mean(per_item_w)),
        "per_item_between": float(np.mean(per_item_b)),
        "per_item_total": float(tot),
        "pooled_within": float(tot * np.sqrt(sw)),
        "pooled_between": float(tot * np.sqrt(1 - sw)),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="arr2026/results")
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    df = load_corpus()
    for r in a.runs:
        p = Path(a.results) / r
        if not (p / "summary.json").exists():
            print(f"{r:18s} MISSING"); continue
        c = components(p, df)
        print(f"{r:18s} tot={c['per_item_total']:.3f} | per-item w={c['per_item_within']:.3f} "
              f"b={c['per_item_between']:.3f} | pooled w={c['pooled_within']:.3f} "
              f"b={c['pooled_between']:.3f}")
