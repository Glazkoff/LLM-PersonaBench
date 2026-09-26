import csv
import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _key():
    with open(ROOT / "data/IPIP-NEO/120/item_key.csv") as f:
        return list(csv.DictReader(f))


_KEYROWS = _key()
FACET_OF_ITEM = {int(r["item"]): r["facet_key"] for r in _KEYROWS}
FACET_NAME = {r["facet_key"]: r["facet"] for r in _KEYROWS}
TEXT_OF_ITEM = {int(r["item"]): r["text"] for r in _KEYROWS}


def _split():
    by = {}
    for i in sorted(FACET_OF_ITEM):
        by.setdefault(FACET_OF_ITEM[i], []).append(i)
    inp = sorted(i for items in by.values() for i in items[:2])
    tgt = sorted(i for items in by.values() for i in items[2:])
    return inp, tgt


ITEM_SPLIT = _split()
ITEM_COLS = [f"i{i}" for i in range(1, 121)]


@dataclass
class Panels:
    cluster: int
    seed: int
    train: pd.DataFrame
    opt: pd.DataFrame
    cal: pd.DataFrame
    eval: pd.DataFrame


def load_corpus(path=None) -> pd.DataFrame:
    return pd.read_csv(path or ROOT / "data/raw/df_ipipneo_120_clusters",
                       usecols=lambda c: c in {"case", "clusters", "sex", "age", "country"} or c in ITEM_COLS)


def excluded_cases(globs) -> set:
    out = set()
    for pat in globs or []:
        for p in glob.glob(str(ROOT / pat), recursive=True):
            d = json.load(open(p))
            for v in d.get("splits", {}).values():
                out.update(int(c) for c in v)
    return out


def make_panels(df, cluster, seed, sizes, exclude=frozenset()) -> Panels:
    pool = df[(df["clusters"] == cluster) & (~df["case"].isin(exclude))]
    pool = pool.dropna(subset=ITEM_COLS)
    rng = np.random.default_rng(seed * 1000 + cluster)
    need = sizes["train"] + sizes["opt"] + sizes["cal"] + sizes["eval"]
    if len(pool) < need:
        raise ValueError(f"cluster {cluster}: {len(pool)} < {need}")
    rows = pool.iloc[rng.permutation(len(pool))[:need]]
    a = sizes["eval"]
    b = a + sizes["cal"]
    c = b + sizes["opt"]
    return Panels(cluster, seed, train=rows.iloc[c:], opt=rows.iloc[b:c], cal=rows.iloc[a:b], eval=rows.iloc[:a])


def answers(df, ids) -> np.ndarray:
    return df[[f"i{i}" for i in ids]].to_numpy(float)


def persona_scores(x_input) -> dict:
    """30 facet scores on 0-100 from the input half (recoded scale: higher = more of the facet)."""
    inp = ITEM_SPLIT[0]
    out = {}
    for fk in sorted(set(FACET_OF_ITEM.values())):
        vals = [x_input[k] for k, i in enumerate(inp) if FACET_OF_ITEM[i] == fk]
        out[fk] = float((np.mean(vals) - 1.0) / 4.0 * 100.0)
    return out
