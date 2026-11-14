# Dependency and access inventory

| Layer | Required dependency | Verification |
|---|---|---|
| Public cell map and launcher tests | Python3.10+ standard library, uv optional | Tested locally; no scientific replay |
| R2 independent numerical replay | Python3.12.14, NumPy2.3.5; vendored auditor+uncertainty pinned by SHA | Existing accepted job27183; not rerun |
| Post-review controls | NumPy2.3.5 plus same auditor and gate | Existing accepted27454; not rerun |
| Gate and restricted input tree | Exact verify_readout.py, successful compare receipt, frozen bundle/config/readout files, original absolute path layout | Launcher checks presence/gate hash; auditor checks full provenance before labels |
| Statistical fitting for proposed pilot | NumPy2.3.5, SciPy1.18.1, pandas3.0.5, sklearn1.7.2 plus exact historical modules/item dictionary | Metadata inventory28016; new FIT-only fits NOT executed |
| Historical LLM readout | openai3.8.0, transformers5.16.1, tokenizers0.23.2; separate vLLM server | Versions in analysis environment verified28016; vLLM absent there, serving environment not attested |
| Manuscript build | pdflatex/BibTeX, ACL style/bibliography, pdfunite | Local multipass build; not a fully containerized TeX environment |

The public source does not contain participant IDs, forecasts, labels or credentials.
The historical full environment lock is preserved, not silently upgraded. Its presence
does not certify complete bitwise inference regeneration. Model revision and the
future pilot's no-retry/gating amendments are documented separately. Installing an
API SDK does not authorize model calls. No install script fetches restricted data.
