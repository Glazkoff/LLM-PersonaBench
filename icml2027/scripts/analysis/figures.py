#!/usr/bin/env python
"""Figures 1-4 from results/aggregates. Colour-blind-safe palette, PDF."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000", "#999999", "#882255"]
ARM_ORDER = ["paraphrase", "random", "bestofb", "ga", "de", "promptbreeder", "mapelites", "opro", "protegi", "gepa", "nsga2"]


def _arms(present):
    return [a for a in ARM_ORDER if a in present] + sorted(set(present) - set(ARM_ORDER))


def fig_delta(c, F, col, lab, name):
    c = c[(c.fitness == "rps_cal") & (~c.excluded)].dropna(subset=[col])
    if c.empty:
        return
    arms, models = _arms(c.arm.unique()), sorted(c.model.unique())
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for k, mod in enumerate(models):
        g = c[c.model == mod].groupby("arm")[col].agg(["mean", "sem", "count"]).reindex(arms)
        x = np.arange(len(arms)) + 0.8 * (k - (len(models) - 1) / 2) / max(len(models), 1)
        ax.errorbar(x, g["mean"], yerr=1.96 * g["sem"].fillna(0), fmt="o", ms=3.5, color=OKABE[k % 10], label=mod, lw=1)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=30, ha="right")
    ax.set_ylabel(lab)
    ax.legend(fontsize=6, ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(F / name)
    plt.close(fig)


def fig_pareto(R, F):
    fs = sorted(Path(R, "cells").glob("*/pareto_front.json"))
    if not fs:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for k, f in enumerate(fs[:16]):
        pts = sorted((p["objectives"] for p in json.loads(f.read_text())), key=lambda o: o[0])
        ax.plot([p[0] for p in pts], [-p[1] for p in pts], "-o", ms=2, color=OKABE[k % 10], lw=0.8)
    ax.set_xlabel("calibrated RPS (lower is better)")
    ax.set_ylabel("VR$_{between}$ (individuation)")
    fig.tight_layout()
    fig.savefig(F / "fig2_pareto.pdf")
    plt.close(fig)


def fig_e2t(A, F):
    p = A / "evaluations_to_threshold.csv"
    if not p.exists():
        return
    e = pd.read_csv(p)
    arms = _arms(e.arm.unique())
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.boxplot([e[e.arm == a].evals.dropna() for a in arms], labels=arms)
    ax.set_ylabel("evaluations to beat seed by 0.005\n(optimisation panel)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(F / "fig3_efficiency.pdf")
    plt.close(fig)


def fig_crossmodel(A, F):
    p = A / "crossmodel_matrix.csv"
    if not p.exists():
        return
    M = pd.read_csv(p, index_col=0)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(M.values, cmap="viridis")
    ax.set_xticks(range(len(M.columns)))
    ax.set_xticklabels(M.columns, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(M.index)))
    ax.set_yticklabels(M.index, fontsize=6)
    ax.set_xlabel("evaluated on")
    ax.set_ylabel("evolved on")
    fig.colorbar(im, ax=ax, label="Δ calibrated RPS on target model")
    fig.tight_layout()
    fig.savefig(F / "fig4_crossmodel.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregates", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A, F = Path(a.aggregates), Path(a.out)
    F.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(A / "tier1_cells.csv") if (A / "tier1_cells.csv").exists() else pd.DataFrame()
    if not c.empty:
        fig_delta(c, F, "delta", "Δ calibrated RPS vs seed persona\n(lower is better)", "fig1_delta_rps.pdf")
        fig_delta(c, F, "mstar_delta", "Δ m* (real answers)", "fig1b_delta_mstar.pdf")
    fig_pareto(A.parent, F)
    fig_e2t(A, F)
    fig_crossmodel(A, F)
    print("figures written to", F)


if __name__ == "__main__":
    main()
