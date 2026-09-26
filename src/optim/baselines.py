"""Non-LLM floors and ceilings on the same frozen panels, plus the 20-redraw split-noise floor."""
import argparse
from pathlib import Path

import numpy as np
import yaml

from src.optim import scoring as S
from src.optim.frozen_eval import floors
from src.optim.panels import FACET_OF_ITEM, ITEM_SPLIT, answers, excluded_cases, load_corpus, make_panels
from src.optim.results import atomic_write_json


def _point(v, n, J):
    P = np.zeros((n, J, 5))
    P[np.arange(n)[:, None], np.arange(J)[None, :], (np.clip(np.rint(v), 1, 5) - 1).astype(int)] = 1
    return P


def _both(P, Y):
    return {"rps": float(np.nanmean(S.rps(P, Y))), "s0": float(np.nanmean(S.s0(P, Y)))}


def baseline_scores(panels, global_train=None):
    tgt = ITEM_SPLIT[1]
    Ytr, Yev = answers(panels.train, tgt), answers(panels.eval, tgt)
    n, J = Yev.shape
    mean = np.broadcast_to(np.nanmean(Ytr, 0), (n, J))
    mode = np.broadcast_to(np.array([np.bincount(Ytr[:, j][~np.isnan(Ytr[:, j])].astype(int), minlength=6)[1:].argmax() + 1
                                     for j in range(J)]), (n, J))
    emp = np.stack([np.array([np.nanmean(Ytr[:, j] == k) for k in range(1, 6)]) for j in range(J)])
    F0 = S.pop_cdf(Ytr)
    fac = {f: np.nanmean(answers(panels.train, [i for i in range(1, 121) if FACET_OF_ITEM[i] == f]))
           for f in set(FACET_OF_ITEM.values())}
    cent = np.array([fac[FACET_OF_ITEM[i]] for i in tgt])
    fl = floors(panels)
    out = {"cluster_mean": _both(_point(mean, n, J), Yev), "cluster_majority": _both(_point(mode, n, J), Yev),
           "empirical_sample": _both(np.broadcast_to(emp, (n, J, 5)), Yev),
           "centroid_deterministic": _both(_point(np.broadcast_to(cent, (n, J)), n, J), Yev),
           "cluster_prior": _both(S.cdf_to_simplex(np.broadcast_to(F0, (n, J, 4))), Yev),
           "human": {"rps": fl["human_rps"], "s0": None}}
    if global_train is not None:
        g = answers(global_train, tgt)
        out["global_mean"] = _both(_point(np.broadcast_to(np.nanmean(g, 0), (n, J)), n, J), Yev)
        out["global_prior"] = _both(S.cdf_to_simplex(np.broadcast_to(S.pop_cdf(g), (n, J, 4))), Yev)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.panels))
    df = load_corpus()
    ex = excluded_cases(cfg["exclude_case_ids_from"])
    gtrain = df.sample(n=20000, random_state=0)
    out = Path(a.out)
    for c in cfg["clusters"]:
        for s in cfg["seeds_headline"]:
            atomic_write_json(out / f"c{c}__s{s}.json",
                              baseline_scores(make_panels(df, c, s, cfg["panel_sizes"], ex), gtrain))
        red = [baseline_scores(make_panels(df, c, 100 + r, cfg["panel_sizes"], ex), gtrain)
               for r in range(1, cfg["baseline_redraws"] + 1)]
        atomic_write_json(out / f"splitnoise_c{c}.json",
                          {k: {"rps_mean": float(np.mean([r[k]["rps"] for r in red])),
                               "rps_sd": float(np.std([r[k]["rps"] for r in red], ddof=1))} for k in red[0]})
        print("baselines cluster", c, flush=True)


if __name__ == "__main__":
    main()
