"""Transfer evaluations for completed cells, run inside a job that serves one model.

ipip300     : persona from the 120 short-form items of the 300 IPIP-NEO-300 respondents
              (data/PAlign/Test-set.json), scored on the 180 items no arm was ever fitted on (H4).
              Respondents are split 150 train (prior, decoder) / 50 calibration (alpha, ridge) / 100 evaluation,
              with a fixed seed. Also prices base and evolved personas in m* on these unseen items, using the
              audit's decoder (h26.fit_decoder) with the 120 short-form answers as the observed pool.
crossmodel  : a cell's evolved genotype evaluated with a DIFFERENT model on the cell's own frozen panels (H6).
crosscluster: a cell's evolved genotype evaluated with the SAME model on another cluster's frozen panels (H6);
              the comparison base is that other cluster's own seed genotype.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.optim import scoring as S
from src.optim.frozen_eval import evaluate_frozen, paired_boot_ci
from src.optim.genotype import seed_genotype
from src.optim.panels import excluded_cases, load_corpus, make_panels
from src.optim.persona import system_prompt
from src.optim.results import atomic_write_json

ROOT = Path(__file__).resolve().parents[2]
K = 5
MSTAR_BUDGETS = [0, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120]


def load_ipip300():
    kx = pd.read_excel(ROOT / "data/PAlign/IPIP-NEO-ItemKey.xls")
    text = {int(r["Full#"]): str(r["Item"]).strip() for _, r in kx.iterrows()}
    facet = {int(r["Full#"]): str(r["Key"]).strip() for _, r in kx.iterrows()}
    rev = {int(r["Full#"]) for _, r in kx.iterrows() if str(r["Sign"]).strip().startswith("-")}
    short = {int(r["Full#"]) for _, r in kx.dropna(subset=["Short#"]).iterrows()}
    rows = json.loads((ROOT / "data/PAlign/Test-set.json").read_text())
    Y = np.array([[float(r[f"i{j}"]) for j in range(1, 301)] for r in rows])
    Y[(Y < 1) | (Y > K)] = np.nan
    return Y, text, facet, rev, sorted(short), sorted(set(range(1, 301)) - short)


def retention(delta_source, delta_transfer):
    if delta_source is None or abs(delta_source) < 1e-6:
        return None
    return float(delta_transfer / delta_source)


def _scores(Yrow, facet, inp):
    out = {}
    for f in sorted({facet[i] for i in inp}):
        v = [Yrow[i - 1] for i in inp if facet[i] == f and not np.isnan(Yrow[i - 1])]
        out[f] = float((np.mean(v) - 1) / 4 * 100) if v else 50.0
    return out


def _interp_mstar(curve, score):
    ms = sorted(curve)
    if score >= curve[ms[0]]:
        return 0.0
    for lo, hi in zip(ms, ms[1:]):
        if curve[lo] >= score >= curve[hi] and curve[lo] > curve[hi]:
            return float(lo + (curve[lo] - score) / (curve[lo] - curve[hi]) * (hi - lo))
    return float(ms[-1])


def ipip300_decoder_curve(Y, inp, tgt, TR, CA, EV, reps=5, seed=20260918):
    sys.path.insert(0, str(ROOT / "arr2026/scripts/hyp"))
    from h25_incremental_screen import onehot, rps as rps_cdf  # noqa: E402
    from h26_answer_equivalent import fit_decoder  # noqa: E402
    Xall, Yall = Y[:, [j - 1 for j in inp]], Y[:, [j - 1 for j in tgt]]
    Xtr, Ytr = Xall[TR], Yall[TR]
    idx = np.concatenate([CA, EV])
    Xs, Yt = Xall[idx], Yall[idx]
    n, nt = Yt.shape
    CALi, EVi = np.arange(len(CA)), np.arange(len(CA), n)
    prior_row = np.stack([np.nanmean(Ytr <= t, axis=0) for t in range(1, K)], axis=1)
    targ = np.nan_to_num(np.concatenate([(Ytr <= t).astype(float) for t in range(1, K)], axis=1), nan=0.0)
    rng = np.random.default_rng(seed)
    curve = {}
    for m in [b for b in MSTAR_BUDGETS if b <= len(inp)]:
        vals = []
        for _ in range(1 if m in (0, len(inp)) else reps):
            if m == 0:
                D = np.broadcast_to(prior_row[None], (n, nt, K - 1))
            else:
                cols = rng.choice(len(inp), m, replace=False)
                D, _ = fit_decoder(onehot(Xtr[:, cols], K), targ, onehot(Xs[:, cols], K), n, nt, K, None, Yt, CALi)
            vals.append(float(np.nanmean(rps_cdf(D[EVi], Yt[EVi], K))))
        curve[m] = float(np.mean(vals))
    return curve


def ipip300(cells, readout, alphas=None):
    Y, text, facet, rev, inp, tgt = load_ipip300()
    alphas = alphas or list(np.round(np.arange(0, 2.01, 0.1), 2))
    perm = np.random.default_rng(20260927).permutation(len(Y))
    TR, CA, EV = perm[:150], perm[150:200], perm[200:]
    Yt = Y[:, [j - 1 for j in tgt]]
    F0 = S.pop_cdf(Yt[TR])
    flip = np.array([j in rev for j in tgt])
    scores = [_scores(Y[i], facet, inp) for i in range(len(Y))]
    cache = {}
    curve_path = ROOT / "icml2027/results/aggregates/ipip300_decoder_curve.json"
    if curve_path.exists():
        curve = {int(k): v for k, v in json.loads(curve_path.read_text()).items()}
    else:
        curve = ipip300_decoder_curve(Y, inp, tgt, TR, CA, EV)
        atomic_write_json(curve_path, curve)
        print("ipip300 decoder curve", curve, flush=True)

    def score_genotype(g):
        key = json.dumps({k: g[k] for k in ("role_definition", "trait_formulations", "facet_formulations",
                                            "critic_formulations")}, sort_keys=True)
        if key in cache:
            return cache[key]
        idx = np.concatenate([CA, EV])
        P, M, _ = readout.beliefs([system_prompt(g, scores[i]) for i in idx], tgt, texts=text)
        P[:, flip, :] = P[:, flip, ::-1]
        Pc, Pe = P[: len(CA)], P[len(CA):]
        a, _ = S.select_alpha(S.cdf(Pc), F0, Yt[CA], alphas)
        per = S.per_respondent(S.cal_cells(S.cdf(Pe), F0, Yt[EV], a, S.cdf(Pc).mean(0)))
        cache[key] = (per, a, float(M.mean()))
        return cache[key]

    prior = float(np.nanmean(S.rps(S.cdf_to_simplex(np.broadcast_to(F0, (len(EV), len(tgt), 4))), Yt[EV])))
    for d in cells:
        out = d / "transfer_ipip300.json"
        if not (d / "eval_frozen.json").exists() or out.exists():
            continue
        c = json.loads((d / "config.json").read_text())["cluster"]
        best = {**seed_genotype(c), **json.loads((d / "best_genotype.json").read_text())["genotype"]}
        pb, ab, mb = score_genotype(seed_genotype(c))
        pe, ae, me = score_genotype(best)
        m, lo, hi = paired_boot_ci(pb, pe, 2000, 20260925)
        bs, es = float(np.nanmean(pb)), float(np.nanmean(pe))
        atomic_write_json(out, {"n_eval": int(len(EV)), "n_targets": len(tgt), "base_rps_cal": bs,
                                "evolved_rps_cal": es, "alpha_base": ab, "alpha_evolved": ae, "delta": m,
                                "ci95": [lo, hi], "prior_rps": prior, "mass_base": mb, "mass_evolved": me,
                                "mstar_base": _interp_mstar(curve, bs), "mstar_evolved": _interp_mstar(curve, es),
                                "mstar_delta": _interp_mstar(curve, es) - _interp_mstar(curve, bs),
                                "per_respondent": {"base": pb.tolist(), "evolved": pe.tolist()}})
        print("ipip300", d.name, f"delta {m:+.4f} [{lo:+.4f},{hi:+.4f}]", flush=True)


def _panels_loader():
    import yaml
    cfg = yaml.safe_load(open(ROOT / "icml2027/configs/panels.yaml"))
    df = load_corpus()
    ex = excluded_cases(cfg["exclude_case_ids_from"])
    cache = {}

    def get(c, s, sizes):
        if (c, s) not in cache:
            cache[(c, s)] = make_panels(df, c, s, sizes, ex)
        return cache[(c, s)]
    return get


def crossmodel(cells, readout, target_slug):
    get = _panels_loader()
    outdir = ROOT / "icml2027/results/crossmodel"
    for d in cells:
        if not (d / "eval_frozen.json").exists():
            continue
        conf = json.loads((d / "config.json").read_text())
        if conf["model_slug"] == target_slug:
            continue
        out = outdir / f"{d.name}__to__{target_slug}.json"
        if out.exists():
            continue
        c, s = conf["cluster"], conf["seed"]
        best = {**seed_genotype(c), **json.loads((d / "best_genotype.json").read_text())["genotype"]}
        ev = evaluate_frozen(get(c, s, conf["panel_sizes"]), readout, seed_genotype(c), best, conf["alphas"], boot_B=2000)
        ev.pop("_beliefs")
        src = json.loads((d / "eval_frozen.json").read_text())
        ev.update(source_cell=d.name, source_model=conf["model_slug"], target_model=target_slug, arm=conf["arm"],
                  cluster=c, seed=s, source_delta=src["delta"]["rps_cal"],
                  retention=retention(src["delta"]["rps_cal"], ev["delta"]["rps_cal"]))
        atomic_write_json(out, ev)
        print("crossmodel", out.name, f"src {src['delta']['rps_cal']:+.4f} -> tgt {ev['delta']['rps_cal']:+.4f}", flush=True)


def crosscluster(cells, readout, slug):
    get = _panels_loader()
    outdir = ROOT / "icml2027/results/crosscluster"
    for d in cells:
        if not (d / "eval_frozen.json").exists():
            continue
        conf = json.loads((d / "config.json").read_text())
        if conf["model_slug"] != slug:
            continue
        c, s = conf["cluster"], conf["seed"]
        best = {**seed_genotype(c), **json.loads((d / "best_genotype.json").read_text())["genotype"]}
        src = json.loads((d / "eval_frozen.json").read_text())
        for c2 in range(4):
            if c2 == c:
                continue
            out = outdir / f"{d.name}__to__c{c2}.json"
            if out.exists():
                continue
            # the other cluster's intensity modifiers and facet names come from its own seed genotype; the
            # transferred wording keeps its trait descriptions, role and critic
            g2 = {**seed_genotype(c2), **{k: best[k] for k in ("role_definition", "trait_formulations",
                                                               "critic_formulations")}}
            ev = evaluate_frozen(get(c2, s, conf["panel_sizes"]), readout, seed_genotype(c2), g2, conf["alphas"],
                                 boot_B=2000)
            ev.pop("_beliefs")
            ev.update(source_cell=d.name, model=slug, arm=conf["arm"], source_cluster=c, target_cluster=c2, seed=s,
                      source_delta=src["delta"]["rps_cal"],
                      retention=retention(src["delta"]["rps_cal"], ev["delta"]["rps_cal"]))
            atomic_write_json(out, ev)
            print("crosscluster", out.name, f"src {src['delta']['rps_cal']:+.4f} -> {ev['delta']['rps_cal']:+.4f}",
                  flush=True)


def select_cells(glob_pat, arms=None, seeds=None, fitness="rps_cal"):
    out = []
    for p in sorted((ROOT / "icml2027/results/cells").glob(glob_pat)):
        if not (p / "config.json").exists():
            continue
        conf = json.loads((p / "config.json").read_text())
        if fitness and conf.get("fitness", "rps_cal") != fitness:
            continue
        if conf.get("mutator") not in (None, "Qwen/Qwen3.8-27B"):
            continue
        if arms and conf["arm"] not in arms:
            continue
        if seeds and conf["seed"] not in seeds:
            continue
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ipip300", "crossmodel", "crosscluster"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--prefill", default="My answer is ")
    ap.add_argument("--cells-glob", default="*")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    a = ap.parse_args()
    from src.optim.readout import VLLMReadout
    readout = VLLMReadout(a.model, prefill=a.prefill, workers=16)
    cells = select_cells(a.cells_glob, a.arms, a.seeds)
    print(len(cells), "cells", flush=True)
    {"ipip300": lambda: ipip300(cells, readout),
     "crossmodel": lambda: crossmodel(cells, readout, a.slug),
     "crosscluster": lambda: crosscluster(cells, readout, a.slug)}[a.mode]()


if __name__ == "__main__":
    sys.exit(main())
