# HBS — consolidated ACL/ARR working manuscript

Editorial staging update 14 November 2026, based on `3933f6173dd537ec867d7377b594b6b6fb02e84c` of branch `codex/hbs-acl-arr-20261111`. This revision includes the 14 November question-selection analysis and S7.

- **[Current paper (PDF)](paper/main.pdf)**; [LaTeX](paper/main.tex).
- **[Complete supplement (PDF)](supplement.pdf)**; [source parts](supplement/).
- [Evidence map and revision checks](REVISION_20261111.md).
- [November follow-up records](followup-20261111/).
- [External-survey aggregates and checker](aggregates/).

The main text combines the historical persona comparison, E3/B1 budget controls,
external R2 stimulus-matched results, statistical-mixture controls and CAL-selection
sensitivity with the November cached-Qwen optimizer diagnostics. Their populations,
estimands and uncertainty remain distinct. S6 contains a synthetic diagnostic only.

This is a working author-review version, not a submission or accepted paper.
Rights, publication overlap and author/ethical declarations remain pending.
The prospective panel has not been recruited and remains NO-GO.

Build `paper/main.tex` using pdflatex, bibtex, then repeated pdflatex passes.
Build S5/S6 with pdflatex twice. The combined supplement concatenates controls,
robustness, extended, external, historical, cached-search and prospective-diagnostic.
The historical S1–S4 PDFs are preserved without reinterpretation as new outcomes.

The 14 November revision adds S7 (question selection) and compact main-text
Methods/Results4.2/Discussion changes, using existing E3 aggregates only.
The combined supplement now appends acquisition.pdf after S6. All twenty curve
budgets,32 random forms and mixed grouped/adaptive controls remain visible.
The numeric cell map now covers263 cells, including226 new acquisition cells;
this checks aggregate-key/rounding/text agreement, not fresh scientific inference.
R2 results and original decisions remain unchanged.

`uv run --no-project python aggregates/audit.py` checks 149 aggregate identities;
it does not reconstruct participant bootstrap intervals. Individual responses and
predictions are not distributed. Hashes of distributed files: `MANIFEST.sha256`.

The 12 November staging additions clarify that smaller-budget saved LLM forecasts
are absent and that CAL reselection does not quantify full-procedure uncertainty.
See `REPRODUCIBILITY.md` for the distinction between public aggregate arithmetic,
existing restricted interval replay, and outstanding portable-replay work;
`CLAIM_EVIDENCE.md` maps the central estimands to their evidence.

The second 12 November packaging step adds `replay/launcher.py`, a 263-cell
source-key/rounding check, pinned historical auditor sources, dependency records,
and a plan-only restricted replay interface. See `replay/README.md`. This is not
a public data release or a new scientific replay. The frozen controls SPEC is
vendored separately from the public explanatory analysis/SPEC.md.
