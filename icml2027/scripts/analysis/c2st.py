#!/usr/bin/env python
"""C2ST-AUC (secondary outcome) for every completed cell, with its style-only rung.

For base and evolved personas: draw one simulated answer per (respondent, item) from the saved belief simplex
(already on the corpus's recoded scale), then train a gradient-boosted classifier to separate simulated from real
answers of the same frozen evaluation respondents (5-fold CV, folded AUC). The style-only rung uses five
content-free scalars per answer vector (mean, SD, extreme-response rate, midpoint rate, longest identical run).
Runs where belief_probs.npz lives (Euler); writes c2st.json next to eval_frozen.json."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.optim.panels import ITEM_SPLIT, answers, excluded_cases, load_corpus, make_panels  # noqa: E402
from src.optim.results import atomic_write_json  # noqa: E402


def style(X):
    def run(r):
        best = cur = 1
        for a, b in zip(r[:-1], r[1:]):
            cur = cur + 1 if a == b else 1
            best = max(best, cur)
        return best
    return np.column_stack([X.mean(1), X.std(1), np.isin(X, [1, 5]).mean(1), (X == 3).mean(1),
                            np.array([run(r) for r in X])])


def auc(A, B, seed):
    X = np.vstack([A, B])
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, random_state=seed), X, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    a = roc_auc_score(y, p)
    return float(max(a, 1 - a))


def sample(P, rng):
    c = P.cumsum(-1)
    u = rng.random(P.shape[:-1])[..., None]
    return (u > c).sum(-1) + 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--panels", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.panels))
    df = load_corpus()
    ex = excluded_cases(cfg["exclude_case_ids_from"])
    cache = {}
    n = 0
    for d in sorted(Path(a.results, "cells").glob("*")):
        if not (d / "belief_probs.npz").exists() or (d / "c2st.json").exists():
            continue
        ev = json.loads((d / "eval_frozen.json").read_text())
        c, s = ev["panel"]["cluster"], ev["panel"]["seed"]
        if (c, s) not in cache:
            cache[(c, s)] = answers(make_panels(df, c, s, cfg["panel_sizes"], ex).eval, ITEM_SPLIT[1])
        H = np.nan_to_num(cache[(c, s)], nan=3.0)
        B = np.load(d / "belief_probs.npz")
        rng = np.random.default_rng(20260925 + s)
        out = {}
        for k in ("base", "evolved"):
            X = sample(B[k].astype(float), rng)
            out[k] = {"c2st_auc": auc(X, H, s), "c2st_style_only": auc(style(X), style(H), s)}
        H2 = H[rng.permutation(len(H))]
        half = len(H) // 2
        out["human_ceiling"] = {"c2st_auc": auc(H2[:half], H2[half:], s)}
        atomic_write_json(d / "c2st.json", out)
        n += 1
    print(n, "cells scored for C2ST")


if __name__ == "__main__":
    main()
