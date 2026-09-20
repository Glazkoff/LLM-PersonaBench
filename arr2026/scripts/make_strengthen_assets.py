"""LaTeX tables and figures for the strengthened paper, from H25/H26 outputs.

Reads whatever is present and says what is missing rather than failing, so it
can be run mid-campaign while cluster shards are still landing.

  H26 json   -> table_answer_equivalent.tex, figures/answer_equivalent.pdf
  H25 csv    -> table_incremental.tex
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CORPUS = {"ipip": "IPIP-NEO-120", "ipip300": "IPIP-NEO-300",
          "sd3": "SD3", "big5": "IPIP-FFM-50", "hexaco": "HEXACO"}


def short_model(m):
    if not m or m == "?":
        return "?"
    return m.split("/")[-1].replace("-Instruct", "").replace("_", ".")


def collect(paths, loader):
    out = []
    for p in paths:
        if p.exists():
            out.extend(loader(p))
    return out


# ------------------------------------------------- Table: answer-equivalent
j_paths = sorted((ROOT / "arr2026/results/h26").glob("*.json")) + \
          sorted((ROOT / "arr2026/results_euler/h26_equiv").glob("*.json"))
rows = collect(j_paths, lambda p: json.loads(p.read_text()))
# Shards overlap (h15h16 subsumes h16; the local pilot subsumes h15), and a
# duplicated cell would be counted twice in every median and every ladder mean.
# Keep the LAST occurrence of each run, which is the most recent scoring.
if rows:
    rows = list({r["run"]: r for r in rows}.values())
    print(f"  {len(rows)} unique cells after de-duplication")
if rows:
    df = pd.DataFrame(rows)
    df["Model"] = df["model"].map(short_model)
    df["Corpus"] = df["corpus"].map(lambda c: CORPUS.get(c, c))
    df["mi"] = pd.to_numeric(df.get("answer_equivalent_interp"), errors="coerce")
    df = df.dropna(subset=["mi"]).sort_values(["Corpus", "llm_calibrated"])

    # ---- compact BODY table: the model ladder, averaged over panels --------
    # The full 72-cell table is appendix material; the body needs the ordering
    # and the spread, not every cell.
    lad = (df.groupby(["Model", "Corpus"])
             .agg(n=("mi", "count"), raw=("llm_raw", "mean"),
                  alpha=("alpha", "mean"), mi=("mi", "mean"))
             .reset_index())
    body = lad[lad.Corpus == "IPIP-NEO-120"].sort_values("mi", ascending=False)
    # The body ladder is for the ORDERING and the spread, not the roster; the
    # full per-cell table is appendix material. Keep the top of the ladder and
    # its zero floor, and mark the elision.
    full_n = len(body)
    if full_n > 10:
        body = pd.concat([body.head(8), body.tail(2)])
    bl = [r"\begin{tabular}{lrrr}", r"\toprule",
          r"Model & raw $S$ & $\alpha^{*}$ & $m^{*}$ (of 60)\\", r"\midrule"]
    for k, (_, r) in enumerate(body.iterrows()):
        if full_n > 10 and k == 8:
            bl.append(r"\midrule \multicolumn{4}{c}{\itshape "
                      f"{full_n - 10} further models, App.~\\ref{{app:equivfull}}"
                      r"}\\ \midrule")
        bl.append(f"{r.Model} & {r.raw:.3f} & {r.alpha:.2f} & "
                  f"\\textbf{{{r.mi:.2f}}}" + r"\\")
    bl += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_ladder.tex").write_text("\n".join(bl))
    print(f"table_ladder.tex  ({len(body)} models on IPIP)")

    # ---- per-corpus summary ------------------------------------------------
    pc = (df.groupby("Corpus")
            .agg(cells=("mi", "count"), raw=("llm_raw", "mean"),
                 cal=("llm_calibrated", "mean"), med=("mi", "median"),
                 mx=("mi", "max")).reset_index())
    pl = [r"\begin{tabular}{lrrrrr}", r"\toprule",
          r"Corpus & cells & raw $S$ & calibrated $S$ & median $m^{*}$ & "
          r"max $m^{*}$\\", r"\midrule"]
    for _, r in pc.iterrows():
        pl.append(f"{r.Corpus} & {int(r.cells)} & {r.raw:.3f} & {r.cal:.3f} & "
                  f"{r.med:.2f} & \\textbf{{{r.mx:.2f}}}" + r"\\")
    pl += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_equiv_by_corpus.tex").write_text("\n".join(pl))
    print(f"table_equiv_by_corpus.tex  ({len(pc)} corpora)")

    # ---- full APPENDIX table ----------------------------------------------
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Model & Corpus & LLM raw & +calib. & $\alpha^{*}$ & "
             r"$m^{*}$ & $m^{*}_{\mathrm{interp}}$\\", r"\midrule"]
    for _, r in df.iterrows():
        mi = r.get("answer_equivalent_interp")
        lines.append(
            f"{r.Model} & {r.Corpus} & {r.llm_raw:.4f} & {r.llm_calibrated:.4f} & "
            f"{r.alpha:.1f} & {r.answer_equivalent} & "
            f"{'--' if mi is None or (isinstance(mi, float) and np.isnan(mi)) else f'{mi:.2f}'}"
            + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_answer_equivalent.tex").write_text("\n".join(lines))
    print(f"table_answer_equivalent.tex  ({len(df)} cells)")

    # ------------------------------------------------ Figure: loss vs budget
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    drawn = 0
    for _, r in df.iterrows():
        c = r.get("curve") or []
        if len(c) < 3:
            continue
        m = [d["m"] for d in c]
        y = [d["loss"] for d in c]
        ax.plot(m, y, marker="o", ms=2.5, lw=1.0, alpha=0.75,
                label=f"{r.Model} / {r.Corpus}")
        ax.axhline(r.llm_calibrated, ls="--", lw=0.8, alpha=0.5,
                   color=ax.lines[-1].get_color())
        drawn += 1
        if drawn >= 4:
            break
    ax.set_xlabel("observed answers given to the decoder, $m$")
    ax.set_ylabel("normalized ordinal RPS")
    ax.set_title("What a persona prompt is worth", fontsize=9)
    ax.legend(fontsize=5.5, frameon=False)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "answer_equivalent.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"figures/answer_equivalent.pdf  ({drawn} curves; dashed = calibrated LLM)")
else:
    print("no H26 results yet -- skipping answer-equivalent assets")

# ------------------------------------------------- Table: incremental screen
c_paths = sorted((ROOT / "arr2026/results/h25").glob("*.csv")) + \
          sorted((ROOT / "arr2026/results_euler/h25_screen").glob("*.csv"))
if c_paths:
    s = pd.concat([pd.read_csv(p) for p in c_paths], ignore_index=True)
    # Local smoke-test CSVs re-score runs the cluster shards already cover;
    # a duplicated cell would inflate the multiplicity denominator and be
    # counted twice in every per-corpus mean.
    before = len(s)
    s = s.drop_duplicates(subset="run", keep="last")
    if len(s) < before:
        print(f"  dropped {before - len(s)} duplicate run rows")
    ok = s[s.status == "ok"].copy()
    if len(ok):
        # MULTIPLICITY. One test per (cell, combiner) is 2-3 x n_cells tests.
        # At a 5% rate chance alone produces dozens of "significant" cells, so
        # the uncorrected counts are not reportable. A paired bootstrap CI that
        # excludes zero corresponds to p < 0.05 two-sided; invert that to a
        # p-value proxy and apply Benjamini-Hochberg across the whole grid.
        import numpy as _np
        fams = [f for f in ("stack", "lin", "gbm") if f"{f}_gain" in ok.columns]
        rec = []
        for f in fams:
            g, lo, hi = ok[f"{f}_gain"], ok[f"{f}_lo"], ok[f"{f}_hi"]
            se = (hi - lo) / (2 * 1.96)
            z = g / se.replace(0, _np.nan)
            from scipy.stats import norm
            rec.append(pd.DataFrame({"fam": f, "idx": ok.index,
                                     "p": 1 - norm.cdf(z)}))
        allp = pd.concat(rec, ignore_index=True).dropna(subset=["p"])
        m = len(allp)
        allp = allp.sort_values("p").reset_index(drop=True)
        allp["bh"] = allp["p"] * m / (allp.index + 1)
        allp["bh"] = allp["bh"][::-1].cummin()[::-1].clip(upper=1.0)
        sig = allp[allp.bh < 0.05]
        print(f"  MULTIPLICITY: {m} one-sided tests across the grid; "
              f"{int((allp.p < 0.05).sum())} nominally significant, "
              f"{len(sig)} survive Benjamini-Hochberg at q<0.05")
        if len(sig):
            for _, r in sig.head(8).iterrows():
                row = ok.loc[r.idx]
                print(f"    {row['run'][:44]:44s} {r.fam:5s} "
                      f"gain {row[f'{r.fam}_gain']:+.5f} q={r.bh:.4f}")
        (OUT / "multiplicity.txt").write_text(
            f"{m} one-sided tests; {int((allp.p<0.05).sum())} nominal; "
            f"{len(sig)} survive BH q<0.05\n")
        ok["Corpus"] = ok["corpus"].map(lambda c: CORPUS.get(c, c))
        g = ok.groupby("Corpus").agg(
            cells=("run", "count"),
            llm=("llm_raw", "mean"), prior=("prior", "mean"),
            dec=("decoder", "mean"), w=("w_star", "mean"),
            lin_sig=("lin_sig", "sum"), gbm_sig=("gbm_sig", "sum"),
            best_lin=("lin_gain", "max"))
        lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
                 r"Corpus & cells & LLM & prior & decoder & $\bar w^{*}$ & "
                 r"sig.\ cells & best $\Delta$\\", r"\midrule"]
        for cor, r in g.iterrows():
            sig = int(r.lin_sig) + int(r.gbm_sig)
            lines.append(f"{cor} & {int(r.cells)} & {r.llm:.3f} & {r.prior:.3f} & "
                         f"{r.dec:.3f} & {r.w:.3f} & {sig}/{int(r.cells)*2} & "
                         f"{r.best_lin:+.4f}" + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (OUT / "table_incremental.tex").write_text("\n".join(lines))
        print(f"table_incremental.tex  ({len(ok)} cells, "
              f"{int(ok.lin_sig.sum())+int(ok.gbm_sig.sum())} significant increments)")
        bad = s[s.status != "ok"]
        if len(bad):
            print(f"  NOTE {len(bad)} runs skipped: "
                  f"{bad.status.value_counts().to_dict()}")
    else:
        print("H25 csv present but no scored rows")
else:
    print("no H25 results yet -- skipping incremental table")
