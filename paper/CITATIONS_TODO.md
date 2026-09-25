# Citation status

## Resolved 2026-09-25

Both previously-absent works are now cited, each verified at source (arXiv
abstract page) rather than from a search snippet:

| key | work | verified |
|---|---|---|
| `hu2024persona` | Hu & Collier, *Quantifying the Persona Effect in LLM Simulations*, ACL 2024 (Vol. 1 Long Papers), 10289–10307, Bangkok | arXiv:2402.10811 |
| `hu2026simbench` | Hu, Baumann, Lupo, Collier, Hovy & Röttger, *SimBench: Benchmarking the Ability of Large Language Models to Simulate Human Behaviors*, ICLR 2026 | arXiv:2510.17516 |

Both render with full author lists in `main.bbl` (18 entries) and are cited in
Related work:

- `hu2024persona` where the paper concedes that persona variables carrying signal
  is already established *and already quantified against supervised regression* —
  under 10% of annotation variance, with persona prompting recovering ~81% of
  what a regression on ground truth achieves. Our question is stated as the
  complementary one: not whether the signal exists, but what it is worth in
  answers.
- `hu2026simbench` beside BehaviorBench, framed as complementary rather than
  competing: standardised group-level evaluation establishes *whether* a
  simulator matches a population, not how many of a person's answers it
  substitutes for.

## Also fixed

`\bibliographystyle{acl_natbib}` was issued both by `acl.sty` and explicitly in
`main.tex`, putting a second `\bibstyle` into `main.aux` so every bibtex run
ended `Illegal, another \bibstyle command ... (There was 1 error message)`.
Harmless in effect — bibtex skips the duplicate and still builds the
bibliography — but it meant a permanent error masked any real one. Pre-existing
since before the strengthening work; the redundant line is removed and bibtex
now runs clean.

## Still to verify before submission

- `binz2025centaur` uses `and others` for authors beyond the first three.
  Replace with the full Nature author list (644(8078):1002–1009, 2025).
- `park2026selfreports` is cited at its 2026 revision; confirm the version and
  whether it now has a venue.
- `huang2025dsa` and `abels2026valuepersonas` are preprints; check for venues.
