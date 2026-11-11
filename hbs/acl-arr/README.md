# HBS — consolidated ACL/ARR working manuscript

Updated 11 November 2026. Branch: `codex/hbs-acl-arr-20261111`.

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

`uv run --no-project python aggregates/audit.py` checks 149 aggregate identities;
it does not reconstruct participant bootstrap intervals. Individual responses and
predictions are not distributed. Hashes of distributed files: `MANIFEST.sha256`.
