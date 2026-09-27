#!/usr/bin/env python
"""Pre-registered hypothesis tests H1-H6 (icml2027/PREREGISTRATION.md section 4), computed only from files under
icml2027/results. Writes aggregates/hypotheses.json and icml2027/RESULTS_BRIEF.md. A hypothesis whose inputs do not
exist yet is reported as 'pending', never guessed."""
import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MUT = "Qwen/Qwen3.8-27B"
BASELINE_ARMS = {"random", "paraphrase", "bestofb"}


def read_csv(path):
    """Missing or column-less CSV -> empty frame (a stage that has not run yet)."""
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def boot_mean(x, B=2000, seed=20260925):
    x = np.asarray([v for v in x if v == v], float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan")), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)])
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def per_resp(R, cell):
    f = R / "cells" / cell / "per_respondent.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return np.array(d["evolved_rps_cal"]) - np.array(d["base_rps_cal"])


def h1(R, cells):
    abl = cells[cells.fitness == "s0"]
    if abl.empty:
        return {"verdict": "pending", "reason": "no S0-fitness ablation cells"}
    pairs, ok = [], 0
    for (m, c), g in abl.groupby(["model", "cluster"]):
        main = cells[(cells.fitness == "rps_cal") & (cells.arm == "ga") & (cells.model == m) & (cells.cluster == c)
                     & (cells.mutator == DEFAULT_MUT)]
        d_s0 = np.concatenate([x for x in (per_resp(R, i) for i in g.cell_id) if x is not None])
        d_rc = [per_resp(R, i) for i in main.cell_id]
        d_rc = np.concatenate([x for x in d_rc if x is not None]) if any(x is not None for x in d_rc) else None
        s0 = boot_mean(d_s0)
        rc = boot_mean(d_rc) if d_rc is not None else (np.nan, np.nan, np.nan)
        hit = bool((s0[1] <= 0 <= s0[2] or s0[1] > 0) and rc[2] < 0)
        ok += hit
        pairs.append({"model": m, "cluster": int(c), "s0_fitness_delta": s0, "rpscal_fitness_delta": rc,
                      "pattern_holds": hit, "s0_fitness_s0_change": float(g.evolved_s0.mean() - g.base_s0.mean())})
    n = len(pairs)
    return {"verdict": "supported" if ok >= 6 and n >= 8 else ("not supported" if n >= 8 else "pending"),
            "pairs_holding": ok, "pairs": n, "detail": pairs, "rule": ">= 6 of 8 (model, cluster) pairs"}


def h2(tests):
    if tests.empty:
        return {"verdict": "pending"}
    out, any_ok = {}, False
    for arm, g in tests[~tests.arm.isin(BASELINE_ARMS)].groupby("arm"):
        r = {}
        for comp in ("random", "paraphrase"):
            gk = g[g.comparator == comp]
            frac = float(((gk.q_bh < 0.05) & (gk.delta < 0)).mean()) if len(gk) else float("nan")
            r[f"frac_sig_better_vs_{comp}"] = frac
            r[f"n_vs_{comp}"] = int(len(gk))
        r["supported"] = bool(r["frac_sig_better_vs_random"] >= 0.5 and r["frac_sig_better_vs_paraphrase"] >= 0.5)
        any_ok |= r["supported"]
        out[arm] = r
    if not out or any(r["n_vs_random"] == 0 or r["n_vs_paraphrase"] == 0 for r in out.values()):
        return {"verdict": "pending", "reason": "random/paraphrase comparison cells not complete", "per_arm": out}
    return {"verdict": "supported" if any_ok else "not supported", "per_arm": out,
            "rule": ">= 50% of an arm's cells significantly better (BH q<0.05) than random AND paraphrase"}


def h3(e2t):
    if e2t.empty:
        return {"verdict": "pending"}
    if "fitness" in e2t:
        e2t = e2t[(e2t.fitness == "rps_cal") & (e2t.mutator.fillna(DEFAULT_MUT) == DEFAULT_MUT)]
    refl = e2t[e2t.arm.isin(["gepa", "protegi"])].evals.dropna()
    evo = e2t[e2t.arm.isin(["ga", "de"])].evals.dropna()
    if len(refl) < 3 or len(evo) < 3:
        return {"verdict": "pending", "n_reflective": int(len(refl)), "n_evolutionary": int(len(evo))}
    p = float(stats.mannwhitneyu(refl, evo, alternative="less").pvalue)
    return {"verdict": "supported" if (refl.median() < evo.median() and p < 0.05) else "not supported",
            "median_reflective": float(refl.median()), "median_evolutionary": float(evo.median()), "p": p,
            "never_reached_reflective": int(e2t[e2t.arm.isin(["gepa", "protegi"])].evals.isna().sum()),
            "never_reached_evolutionary": int(e2t[e2t.arm.isin(["ga", "de"])].evals.isna().sum()),
            "test": "one-sided Mann-Whitney (pre-registration names Wilcoxon; cells are not paired across arms)"}


def h4(cells):
    c = cells[(cells.fitness == "rps_cal") & cells.ipip300_mstar_delta.notna() & (cells.mutator == DEFAULT_MUT)]
    if c.empty:
        return {"verdict": "pending", "reason": "IPIP-NEO-300 transfer not run yet"}
    m = boot_mean(c.ipip300_mstar_delta)
    wide = c.pivot_table(index=["arm", "model", "cluster"], columns="seed", values="ipip300_mstar_delta")
    rhos = []
    cols = list(wide.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            x = wide[[cols[i], cols[j]]].dropna()
            if len(x) > 5:
                rhos.append(stats.spearmanr(x.iloc[:, 0], x.iloc[:, 1]).correlation)
    rho = float(np.mean(rhos)) if rhos else float("nan")
    return {"verdict": "supported" if (m[1] > 0 and rho > 0.5) else "not supported",
            "pooled_delta_mstar": m, "cross_panel_spearman": rho, "n_cells": int(len(c)),
            "pooled_delta_rps_cal_ipip300": boot_mean(c.ipip300_delta)}


def h5(R, cells):
    e5 = cells[cells.arm == "nsga2"]
    if e5.empty:
        return {"verdict": "pending"}
    rows, hits = [], 0
    for _, r in e5.iterrows():
        ga = cells[(cells.arm == "ga") & (cells.fitness == "rps_cal") & (cells.model == r.model)
                   & (cells.cluster == r.cluster) & (cells.seed == r.seed) & (cells.mutator == DEFAULT_MUT)]
        if ga.empty:
            continue
        g = ga.iloc[0]
        dom = bool(r.evolved_rps_cal <= g.evolved_rps_cal and r.vr_between_evolved >= g.vr_between_evolved)
        hits += dom
        rows.append({"cell": r.cell_id, "nsga2_rps": r.evolved_rps_cal, "ga_rps": g.evolved_rps_cal,
                     "nsga2_vr": r.vr_between_evolved, "ga_vr": g.vr_between_evolved, "dominates": dom})
    frac = hits / len(rows) if rows else float("nan")
    return {"verdict": "supported" if frac >= 0.5 else "not supported", "frac_cells_dominating": frac,
            "n": len(rows), "detail": rows,
            "note": "dominance checked on the frozen panel between the NSGA-II knee and the GA best genotype"}


def h6(A):
    cm, cc = read_csv(A / "crossmodel.csv"), read_csv(A / "crosscluster.csv")
    if cm.empty or cc.empty:
        return {"verdict": "pending"}
    rm, rc = cm.retention.dropna().to_numpy(), cc.retention.dropna().to_numpy()
    rng = np.random.default_rng(20260925)
    diffs = [rng.choice(rm, len(rm)).mean() - rng.choice(rc, len(rc)).mean() for _ in range(2000)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"verdict": "supported" if hi < 0 else "not supported", "retention_crossmodel": boot_mean(rm),
            "retention_crosscluster": boot_mean(rc), "difference_ci95": [float(lo), float(hi)]}


def fmt(ci):
    return f"{ci[0]:+.4f} [{ci[1]:+.4f}, {ci[2]:+.4f}]" if isinstance(ci, (list, tuple)) else str(ci)


def brief(res, cells, A, out):
    main = cells[(cells.fitness == "rps_cal") & (cells.mutator == DEFAULT_MUT)]
    L = [f"# Results brief — icml2027 (generated {date.today().isoformat()} by scripts/analysis/hypotheses.py)", "",
         "Every number below is computed from files under `icml2027/results/`. Pending means the inputs do not exist yet.",
         "", f"Completed cells: {len(cells)} (Tier-1 main-fitness: {len(main)}).", "",
         "## Tier-1 by arm (mean over completed cells)", "",
         "| arm | cells | Δ calibrated RPS | cells with CI < 0 | Δ m* (answers) | evolved vs prior | evolved vs wrong-persona |",
         "|---|---|---|---|---|---|---|"]
    for arm, g in main.groupby("arm"):
        L.append(f"| {arm} | {len(g)} | {fmt(boot_mean(g.delta))} | {(g.ci_hi < 0).sum()}/{len(g)} | "
                 f"{g.mstar_delta.mean():+.2f} | {(g.evolved_rps_cal - g.floor_cluster_prior_rps).mean():+.4f} | "
                 f"{(g.evolved_rps_cal - g.wrong_persona).mean():+.4f} |")
    L += ["", "## Pre-registered hypotheses", ""]
    names = {"H1": "improper S0 fitness vs calibrated-RPS fitness", "H2": "selection-pressure arms beat random and paraphrase",
             "H3": "reflective arms reach threshold with fewer evaluations", "H4": "optimised personas raise m* on unseen IPIP-300 items",
             "H5": "multi-objective search yields a non-degenerate front", "H6": "cross-model transfer weaker than cross-cluster"}
    for k in ("H1", "H2", "H3", "H4", "H5", "H6"):
        r = res[k]
        extra = {kk: v for kk, v in r.items() if kk not in ("verdict", "detail", "per_arm", "pairs")}
        L.append(f"- **{k} ({names[k]}): {r['verdict']}.** " + "; ".join(f"{kk} = {fmt(v) if isinstance(v, tuple) else v}"
                                                                          for kk, v in extra.items()))
    Path(out).write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "icml2027/results"))
    ap.add_argument("--brief", default=str(ROOT / "icml2027/RESULTS_BRIEF.md"))
    a = ap.parse_args()
    R = Path(a.results)
    A = R / "aggregates"
    cells = pd.read_csv(A / "tier1_cells.csv")
    if "mutator" not in cells:
        cells["mutator"] = DEFAULT_MUT
    cells["mutator"] = cells["mutator"].fillna(DEFAULT_MUT)
    for col in ("ipip300_mstar_delta", "ipip300_delta"):
        if col not in cells:
            cells[col] = np.nan
    tests, e2t = read_csv(A / "tests_bh.csv"), read_csv(A / "evaluations_to_threshold.csv")
    res = {"H1": h1(R, cells), "H2": h2(tests), "H3": h3(e2t), "H4": h4(cells), "H5": h5(R, cells), "H6": h6(A)}
    (A / "hypotheses.json").write_text(json.dumps(res, indent=1, default=float))
    brief(res, cells, A, a.brief)
    for k, v in res.items():
        print(k, v["verdict"])


if __name__ == "__main__":
    main()
