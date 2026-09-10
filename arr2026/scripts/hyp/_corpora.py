r"""One loader for every instrument, with a uniform contract.

    load(name) -> Corpus(Y, ids, text, scale, rev, n_input, recoded, K)

  Y         (respondents, items) answers, NaN where missing
  ids       item ids, index-aligned with Y's columns
  text      id -> the literal item as the respondent saw it
  scale     id -> the facet/trait it belongs to
  rev       ids that are reverse-keyed
  n_input   how many items per scale condition the persona; the rest are targets
  recoded   whether Y is stored already reverse-recoded
  K         number of response levels (5 for IPIP/SD3/BIG5, 7 for HEXACO)

`recoded` and `rev` come from _orientation_detect for corpora with no published
key, and are cross-checked against the published key where one exists. Getting
this wrong is invisible in any variance statistic and wrong in every
correlation and absolute-error score, so the loader refuses to guess silently:
check_orientation() reports agreement and disagreement per corpus.
"""
from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _orientation_detect import detect_recoded, reverse_keyed  # noqa: E402


@dataclass
class Corpus:
    name: str
    Y: np.ndarray
    ids: list
    text: dict
    scale: dict
    rev: set
    n_input: int
    recoded: bool
    K: int

    def scale_list(self):
        return [self.scale[i] for i in self.ids]


def _ipip():
    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")
    key = pd.read_csv(ROOT / "data/IPIP-NEO/120/item_key.csv")
    text = {int(r.item): r.text for r in key.itertuples()}
    scale = {int(r.item): r.facet_key for r in key.itertuples()}
    rev = {int(r.item) for r in key.itertuples() if str(r.reverse).lower() == "true"}
    cols = [f"i{i}" for i in range(1, 121)]
    return Corpus("ipip", df[cols].to_numpy(float), list(range(1, 121)),
                  text, scale, rev, 2, True, 5)


def _sd3():
    rows = list(csv.DictReader(open(ROOT / "data/PAlign/Dark-Triad.csv"), delimiter="\t"))
    keys = [f"{t}{i}" for t in "MNP" for i in range(1, 10)]
    Y = np.array([[float(r[k]) for k in keys] for r in rows])
    Y = Y[(Y >= 1).all(axis=1)]
    kx = pd.read_excel(ROOT / "data/PAlign/dark_triad-ItemKey.xls")
    text = {i + 1: str(r.Item).strip() for i, r in enumerate(kx.itertuples())}
    scale = {i + 1: keys[i][0] for i in range(len(keys))}
    rev = {i + 1 for i, r in enumerate(kx.itertuples()) if str(r.Sign).strip() == "-"}
    return Corpus("sd3", Y, list(range(1, 28)), text, scale, rev, 4, False, 5)


def _codebook_text(path, pattern):
    out = {}
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = re.match(rf"^({pattern})[\t ]+(.+?)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _openpsy(name, folder, pattern, scale_of, K, n_input):
    d = ROOT / "data/openpsychometrics" / folder
    df = pd.read_csv(d / "data.csv", sep=None, engine="python")
    cols = [c for c in df.columns if re.fullmatch(pattern, c)]
    Y = df[cols].to_numpy(float)
    Y[(Y < 1) | (Y > K)] = np.nan                 # 0 encodes "missed"
    txt = _codebook_text(d / "codebook.txt", pattern)
    ids = list(range(1, len(cols) + 1))
    text = {i: txt.get(c, c) for i, c in zip(ids, cols)}
    scale = {i: scale_of(c) for i, c in zip(ids, cols)}
    rec, _, _ = detect_recoded(Y, [scale_of(c) for c in cols])
    rev = {ids[j] for j in reverse_keyed(Y, [scale_of(c) for c in cols])}
    return Corpus(name, Y, ids, text, scale, rev, n_input, rec, K)


def _hexaco():
    return _openpsy("hexaco", "HEXACO", r"[HEXACO][A-Za-z]{3,4}\d{1,2}",
                    lambda c: re.sub(r"\d+$", "", c), 7, 5)


def _big5():
    return _openpsy("big5", "BIG5", r"[ENACO]\d{1,2}", lambda c: c[0], 5, 5)


LOADERS = {"ipip": _ipip, "sd3": _sd3, "hexaco": _hexaco, "big5": _big5}


def load(name: str) -> Corpus:
    return LOADERS[name]()


def check_orientation(c: Corpus):
    """Compare the published key (if any) with what the data says."""
    det_rec, frac_neg, mean_r = detect_recoded(c.Y, c.scale_list())
    det_rev = {c.ids[j] for j in reverse_keyed(c.Y, c.scale_list())}
    agree_storage = det_rec == c.recoded
    # IPIP and SD3 ship a published key, so comparing it against the detector is
    # an independent check. HEXACO and BIG5 have no key here and their `rev` IS
    # the detector's output, so an overlap of 1.0 there is circular and must not
    # be reported as if it confirmed anything.
    has_key = c.name in ("ipip", "sd3")
    overlap = (len(det_rev & c.rev) / max(len(c.rev), 1)
               if has_key and not c.recoded else None)
    return dict(corpus=c.name, K=c.K, n=int(c.Y.shape[0]), items=len(c.ids),
                scales=len(set(c.scale.values())), declared_recoded=c.recoded,
                detected_recoded=det_rec, agree=agree_storage,
                frac_negative=round(frac_neg, 3), mean_r=round(mean_r, 3),
                reverse_source="published key" if has_key else "detected (no key)",
                n_reverse=len(c.rev),
                key_vs_detected=None if overlap is None else round(overlap, 3))
