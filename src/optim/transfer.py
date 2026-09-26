"""Transfer evaluations for completed cells, run inside a job that serves one model.

ipip300   : persona from the 120 short-form items of the 300 IPIP-NEO-300 respondents
            (data/PAlign/Test-set.json), scored on the 180 items no arm was ever fitted on (H4).
            Respondents are split 150 train (prior) / 50 calibration (alpha) / 100 evaluation, fixed seed.
crossmodel: a cell's evolved genotype evaluated with a DIFFERENT model on the cell's own frozen panels (H6).
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


def ipip300(cells, readout, out_name="transfer_ipip300.json", alphas=None):
    Y, text, facet, rev, inp, tgt = load_ipip300()
    alphas = alphas or list(np.round(np.arange(0, 2.01, 0.1), 2))
    perm = np.random.default_rng(20260927).permutation(len(Y))
    TR, CA, EV = perm[:150], perm[150:200], perm[200:]
    Yt = Y[:, [j - 1 for j in tgt]]
    F0 = S.pop_cdf(Yt[TR])
    flip = np.array([j in rev for j in tgt])
    scores = [_scores(Y[i], facet, inp) for i in range(len(Y))]
    cache = {}

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

    for d in cells:
        ev_path = d / "eval_frozen.json"
        out = d / out_name
        if not ev_path.exists() or out.exists():
            continue
        c = json.loads((d / "config.json").read_text())["cluster"]
        best = json.loads((d / "best_genotype.json").read_text())["genotype"]
        best = {**seed_genotype(c), **best}
        pb, ab, mb = score_genotype(seed_genotype(c))
        pe, ae, me = score_genotype(best)
        m, lo, hi = paired_boot_ci(pb, pe, 2000, 20260925)
        prior = float(np.nanmean(S.rps(S.cdf_to_simplex(np.broadcast_to(F0, (len(EV), len(tgt), 4))), Yt[EV])))
        atomic_write_json(out, {"n_eval": int(len(EV)), "n_targets": len(tgt), "base_rps_cal": float(np.nanmean(pb)),
                                "evolved_rps_cal": float(np.nanmean(pe)), "alpha_base": ab, "alpha_evolved": ae,
                                "delta": m, "ci95": [lo, hi], "prior_rps": prior, "mass_base": mb, "mass_evolved": me,
                                "per_respondent": {"base": pb.tolist(), "evolved": pe.tolist()}})
        print("ipip300", d.name, f"delta {m:+.4f} [{lo:+.4f},{hi:+.4f}]", flush=True)


def crossmodel(cells, readout, target_slug, panels_yaml="icml2027/configs/panels.yaml"):
    import yaml
    cfg = yaml.safe_load(open(ROOT / panels_yaml))
    df = load_corpus()
    ex = excluded_cases(cfg["exclude_case_ids_from"])
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
        panels = make_panels(df, c, s, conf["panel_sizes"], ex)
        best = {**seed_genotype(c), **json.loads((d / "best_genotype.json").read_text())["genotype"]}
        ev = evaluate_frozen(panels, readout, seed_genotype(c), best, conf["alphas"], boot_B=2000)
        ev.pop("_beliefs")
        src = json.loads((d / "eval_frozen.json").read_text())
        ev["source_cell"] = d.name
        ev["source_delta"] = src["delta"]["rps_cal"]
        ev["target_model"] = target_slug
        ev["retention"] = retention(src["delta"]["rps_cal"], ev["delta"]["rps_cal"])
        atomic_write_json(out, ev)
        print("crossmodel", out.name, f"src {src['delta']['rps_cal']:+.4f} -> tgt {ev['delta']['rps_cal']:+.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ipip300", "crossmodel"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--prefill", default="My answer is ")
    ap.add_argument("--cells-glob", required=True)
    a = ap.parse_args()
    from src.optim.readout import VLLMReadout
    readout = VLLMReadout(a.model, prefill=a.prefill, workers=16)
    cells = sorted(p for p in (ROOT / "icml2027/results/cells").glob(a.cells_glob) if p.is_dir())
    print(len(cells), "cells", flush=True)
    if a.mode == "ipip300":
        ipip300(cells, readout)
    else:
        crossmodel(cells, readout, a.slug)


if __name__ == "__main__":
    sys.exit(main())
