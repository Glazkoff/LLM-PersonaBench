# HBS / A16 — current ACL/ARR manuscript

Latest manuscript revision: 10 November 2026; uploaded 11 November 2026.

- **[Read the paper (PDF)](paper/main.pdf)** — 8 pages, ACL review format.
- [Editable main text (LaTeX)](paper/main.tex).
- [Complete supplement (PDF)](supplement.pdf) — 49 pages.
- [Supplement source parts](supplement/).
- [Aggregate results and checker](aggregates/).

This is the current author-review version, not a conference submission or an accepted paper. It incorporates the statistical-mixture controls and CAL-selection sensitivity from the second Claude review. Granite's fixed-mixture component gain does not establish superiority over strong statistical mixing or a stable persona-specific benefit. Negative and mixed findings remain included.

## Reproduction and scope
From `paper/`: run pdflatex main, bibtex main, then pdflatex main twice. The line-number hook supports outer-margin numbering with lineno 5.7; ACL geometry is unchanged.

`uv run --no-project python aggregates/audit.py` checks aggregate arithmetic (149 checks); it does not reconstruct participant bootstrap intervals. `analysis/controls.py` requires the restricted dataset and audit dependencies described in its README. Individual responses and predictions are not distributed.

Combined supplement page map: S1 1–3, S1b 4–5, S2 6–19, S3 20–24, S4 25–49. The main paper and S1 qualify earlier interpretations in the historical materials.

Publication overlap, rights, ethical and author declarations remain pending before submission. A17 recruitment and study execution remain NO-GO. No new outcomes are introduced by this upload. This GitHub publication does not authorize article submission or a new experiment.

SHA-256 hashes are in MANIFEST.sha256. The prior working checkout branch was `codex/hbs-aamas-comparison`; this self-contained revision is on `codex/hbs-acl-arr-20261111`.
