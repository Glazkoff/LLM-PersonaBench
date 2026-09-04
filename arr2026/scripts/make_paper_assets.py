"""Generate LaTeX tables + figures for the ARR paper from the result files."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "arr2026/results"
OUT = ROOT.parent / "paper"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

PRETTY = {"gigchat3": "GigaChat3-10B", "gpt4_mini": "GPT-4.1-mini",
          "gpt4_nano": "GPT-4.1-nano", "grok4.1_fast": "Grok-4.1-fast",
          "qwen3": "Qwen3-235B"}

# ---------------------------------------------------------- Table 1: head-to-head
h = pd.read_csv(RES / "e2b/head_to_head.csv")
h = h[h.llm_after.notna()].copy()
h["model"] = h.model.map(PRETTY).fillna(h.model)
piv = h.pivot_table(index="model", columns="cluster", values="llm_after")
base = h.pivot_table(index="model", columns="cluster", values="base_cluster_mean").iloc[0]

lines = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"Model & \multicolumn{4}{c}{Cluster} & Mean\\",
         r"\cmidrule(lr){2-5}",
         r" & 0 & 1 & 2 & 3 & \\", r"\midrule"]
for m, row in piv.iterrows():
    lines.append(f"{m} & " + " & ".join(f"{row[c]:.3f}" for c in [0, 1, 2, 3])
                 + f" & {row.mean():.3f}" + r"\\")
lines += [r"\midrule",
          r"\textit{Cluster mean} (constant) & " +
          " & ".join(f"\\textbf{{{base[c]:.3f}}}" for c in [0, 1, 2, 3]) +
          f" & \\textbf{{{base.mean():.3f}}}" + r"\\",
          r"\textit{Real human, same cluster} & " +
          " & ".join(f"{h[h.cluster==c].base_train_user_copy.mean():.3f}" for c in [0,1,2,3]) +
          f" & {h.base_train_user_copy.mean():.3f}" + r"\\",
          r"\bottomrule", r"\end{tabular}"]
(OUT / "table_headtohead.tex").write_text("\n".join(lines))

# ---------------------------------------------------------- Table 2: metric suite
e3 = pd.read_csv(RES / "e3/metric_suite.csv")
a = e3[e3.stage == "after"]
rows = [
    ("C2ST-AUC $\\downarrow$", a.c2st_model.mean(), a.c2st_human_ceiling.mean(), a.c2st_constant_floor.mean()),
    ("DF $\\uparrow$", a.df_model.mean(), a.df_human_ceiling.mean(), a.df_constant_floor.mean()),
    ("VR (1.0 ideal)", a.vr_model.mean(), a.vr_human_ceiling.mean(), a.vr_constant_floor.mean()),
    ("$S_{\\mathrm{answer}}$ $\\uparrow$ (old)", a.s_answer_model.mean(), a.s_answer_human_ceiling.mean(), a.s_answer_constant_floor.mean()),
]
t2 = [r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{lccc}", r"\toprule",
      r"Metric & LLM & Human & Const.\\",
      r" & & ceiling & floor\\", r"\midrule"]
for name, m, hc, cf in rows:
    star = r"\;$\dagger$" if name.startswith("$S_") else ""
    t2.append(f"{name} & {m:.3f} & {hc:.3f} & {cf:.3f}{star}" + r"\\")
t2 += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_metricsuite.tex").write_text("\n".join(t2))

# ---------------------------------------------------------- Table 3: proposition
p = pd.read_csv(RES / "e3/proposition.csv")
t3 = [r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{lccccc}", r"\toprule",
      r"Clu. & $n$ & MAD & GMD & $S_{\mathrm{const}}$ & $S_{\mathrm{faith}}$\\", r"\midrule"]
for _, r in p.iterrows():
    t3.append(f"{int(r.cluster)} & {int(r.n):,} & {r.MAD_mean:.3f} & {r.GMD_mean:.3f} & "
              f"{r.score_constant:.4f} & {r.score_faithful:.4f}" + r"\\")
t3 += [r"\midrule",
       f"Mean & & {p.MAD_mean.mean():.3f} & {p.GMD_mean.mean():.3f} & "
       f"\\textbf{{{p.score_constant.mean():.4f}}} & {p.score_faithful.mean():.4f}" + r"\\",
       r"\bottomrule", r"\end{tabular}"]
(OUT / "table_proposition.tex").write_text("\n".join(t3))

# ---------------------------------------------------------- Figure 1: the inversion
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2))
lbl = ["LLM\n(evolved)", "Real human\n(same cluster)", "Constant\n(cluster mean)"]
cols = ["#c0392b", "#2c7fb8", "#7f7f7f"]

old = [a.s_answer_model.mean(), a.s_answer_human_ceiling.mean(), a.s_answer_constant_floor.mean()]
b0 = ax[0].bar(lbl, old, color=cols, width=.62)
ax[0].set_title(r"Old metric $S_{\mathrm{answer}}$  (higher = better)", fontsize=9, pad=16)
ax[0].set_ylim(0.60, 0.88)
ax[0].set_ylabel("score", fontsize=8.5)
for r, v in zip(b0, old):
    ax[0].text(r.get_x()+r.get_width()/2, v+.005, f"{v:.3f}", ha="center", fontsize=8)
ax[0].annotate("", xy=(2, 0.845), xytext=(1, 0.845),
               arrowprops=dict(arrowstyle="<-", color="#c0392b", lw=1.2))
ax[0].text(1.5, 0.852, "constant outscores\nthe real human", ha="center",
           fontsize=8, color="#c0392b", fontweight="bold")

new = [a.c2st_model.mean(), a.c2st_human_ceiling.mean(), a.c2st_constant_floor.mean()]
b1 = ax[1].bar(lbl, new, color=cols, width=.62)
ax[1].set_title(r"Proposed C2ST-AUC  (0.5 = ideal)", fontsize=9, pad=16)
ax[1].set_ylim(0.40, 1.14)
ax[1].axhline(0.5, ls=":", lw=1.1, c="k")
ax[1].text(2.48, 0.515, "ideal (0.5)", fontsize=7.5, ha="right", va="bottom")
for r, v in zip(b1, new):
    ax[1].text(r.get_x()+r.get_width()/2, v+.012, f"{v:.3f}", ha="center", fontsize=8)
ax[1].annotate("", xy=(1, 1.02), xytext=(2, 1.02),
               arrowprops=dict(arrowstyle="<-", color="#2c7fb8", lw=1.2))
ax[1].text(1.5, 1.045, "human is near chance,\nconstant is separable", ha="center",
           fontsize=8, color="#2c7fb8", fontweight="bold")
for x in ax:
    x.tick_params(labelsize=8); x.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(FIG / "inversion.pdf", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------- Figure 2: k selection
try:
    k = pd.read_csv(RES / "e1/k_selection.csv")
    z = k[k.preprocessing == "zscore"]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
    for i, (col, name, best) in enumerate([
            ("silhouette", "Silhouette $\\uparrow$", "max"),
            ("davies_bouldin", "Davies--Bouldin $\\downarrow$", "min"),
            ("ari_vs_shipped", "ARI vs.\\ published labels", None)]):
        ax[i].plot(z.k, z[col], "o-", ms=3, lw=1.2, color="#2c7fb8")
        ax[i].axvline(4, color="#c0392b", ls="--", lw=1)
        ax[i].set_title(name, fontsize=8); ax[i].set_xlabel("$k$", fontsize=8)
        ax[i].tick_params(labelsize=7); ax[i].spines[["top","right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(FIG / "kselection.pdf", bbox_inches="tight"); plt.close()
    print("figure kselection.pdf written")
except FileNotFoundError:
    print("E1 not finished; kselection figure skipped")

# ---------------------------------------------------------- Figure 3: variance
fig, ax = plt.subplots(figsize=(3.4, 2.4))
vm = a.groupby("model").vr_model.mean().rename(index=PRETTY).sort_values()
ax.barh(vm.index, vm.values, color="#c0392b")
ax.axvline(1.0, ls="--", c="#2c7fb8", lw=1.2)
ax.text(1.02, -0.4, "human", color="#2c7fb8", fontsize=7)
ax.set_xlabel("variance ratio (model / human)", fontsize=8)
ax.tick_params(labelsize=7); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig(FIG / "variance.pdf", bbox_inches="tight"); plt.close()

print("assets written to", OUT)
