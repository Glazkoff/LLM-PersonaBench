# Citation status — all items resolved 2026-09-25

Every entry verified at source (arXiv abstract page, or PMC for the paywalled
Nature article) rather than from a search snippet. `refs.bib` holds 18 entries;
`main.bbl` renders 17 (one is cited only indirectly). bibtex runs clean and the
build has zero LaTeX warnings.

## Added

| key | work | source checked |
|---|---|---|
| `hu2024persona` | Hu & Collier, *Quantifying the Persona Effect in LLM Simulations*, ACL 2024 (Vol. 1 Long), 10289–10307 | arXiv:2402.10811 |
| `hu2026simbench` | Hu, Baumann, Lupo, Collier, Hovy & Röttger, *SimBench*, ICLR 2026 | arXiv:2510.17516 |

## Corrected

| key | was | now |
|---|---|---|
| `binz2025centaur` | `and others` after the third author | full 40-author Nature list + DOI 10.1038/s41586-025-09215-4, via PMC12390832 (nature.com redirects to an auth wall) |
| `abels2026valuepersonas` | bare `@article` preprint | `@inproceedings`, VALE workshop at IJCAI-ECAI 2026 — it *does* have a venue |
| `huang2025dsa` → `huang2026dsa` | year 2025 | v2, April 2026; still no venue. Key renamed so it stops disagreeing with the version cited |
| `park2026selfreports` | no version | v3, June 2026; confirmed still venueless |

## Removed — a duplicate of one work under two keys

`park2024generative` ("Generative Agent Simulations of 1,000 People", 2024) and
`park2026selfreports` ("LLM Agents Grounded in Self-Reports…", 2026) are **the
same arXiv paper**, 2411.10109, retitled between v1 and v3. The bibliography
listed both, and Related work cited them in two different places as if they were
independent works.

Merged into `park2026selfreports` (the current title), and the two prose mentions
rewritten so the work is introduced once and referred back to, rather than
double-cited. This was half pre-existing and half introduced when the second
entry was added without checking the eprint of the first — the lesson being to
match on eprint ID, not on title, before adding a reference.

## Nothing outstanding

The three preprints without venues (`huang2026dsa`, `park2026selfreports`, and
the `choi2026beyondmean` workshop paper) are cited as preprints deliberately and
correctly; re-check them if the submission slips past their publication.
