# External transfer adapter status (4 October 2026)

The CPU adapter is `aamas2027.external`. It prepares **Twin-2K-500** from the official question catalog and numeric human-response CSVs. It does not download or parse `full_persona`, `persona_summary`, or `wave4_Q_wave1_3_A` as a model input. The [dataset card](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500) says full personas contain wave 4 answers, while the wave 1–3 CSV also contains answers to some repeated wave 4 questions. The [official catalog README](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500/blob/main/question_catalog_and_human_response_csv/question_catalog_README.md) describes the `csv_columns`, question types, selectors, and their expansion. The [official code](https://github.com/tianyipeng-lab/Digital-Twin-Simulation) is the primary simulation reference.

## What is implemented

On Euler, in a finite CPU Slurm allocation, use the [verified dataset commit](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500/commit/f883165a3026fde855dfd448e0cd16443ab257b6) and run:

```sh
uv run -m aamas2027.external download --revision f883165a3026fde855dfd448e0cd16443ab257b6 --out <PRIVATE_DATA_DIR>
uv run -m aamas2027.external readiness --data <PRIVATE_DATA_DIR> --selection <FROZEN_SELECTION_JSON>
uv run -m aamas2027.external prepare --data <PRIVATE_DATA_DIR> --selection <FROZEN_SELECTION_JSON> --out <PRIVATE_OUTPUT_DIR>
```

The download obtains only `question_catalog.json`, `wave1_3_response.csv`, and `wave4_response.csv` from the pinned revision and writes SHA-256 receipts. The `prepare` command writes a manifest with namespaced `twin2k:<pid>` IDs, deterministic 1000/200/200/remainder fit/dev/cal/test split, a typed target catalog, label denominators, allowed history columns, and SHA-256 hashes. `rows.jsonl` contains only whitelisted wave 1–3 history and wave 4 targets. Keep this file private on Euler; it contains respondent answers. No full human dataset was downloaded or processed on the Mac for this work.

Selection JSON must be frozen before model outcomes are read. Its structure is:

```json
{
  "source": "catalog_csv",
  "overlap_audit": true,
  "families": ["economic_preferences", "heuristics_cognitive_bias"],
  "equivalent_question_ids": [],
  "targets": [
    {"column": "<CATALOG_CSV_COLUMN>", "kind": "nominal", "family": "economic_preferences"},
    {"column": "<CATALOG_CSV_COLUMN>", "kind": "ordinal", "family": "heuristics_cognitive_bias"}
  ]
}
```

These are schema placeholders, **not selected endpoints**. Before setting `overlap_audit: true`, inspect exact question text, response options, blocks, and all equivalent or paraphrased wave 1–3 questions against the official codebook and record that audit in the selection artifact. List equivalent QuestionIDs in `equivalent_question_ids`. The adapter automatically excludes every selected target's entire QuestionID from input history; it also excludes every declared equivalent QuestionID. Unknown history columns and unknown equivalent IDs fail preparation. This protects exact repeats; paraphrase detection requires the documented codebook review. The two prespecified families must both be present even if one performs poorly. No separate human approval is required by the adapter.

Only documented Matrix/Likert columns can be declared `ordinal`, with RPS. Single-select MC can be declared `nominal` (or `binary` with two options), with Brier. The adapter rejects sliders, text, multi-select, and ambiguous scales. No arbitrary nominal option index becomes an ordered target. Continuous targets remain gated until a distributional readout and CRPS are implemented.

The data gate is **open only after** an actual pinned download, selected-endpoint audit, and successful `prepare` on Euler. Source browsing established the catalog format, but no full dataset or endpoint manifest has been audited here. The adapter itself is not a forecast runner, so model readout and matched-history statistical comparators remain separate gates.

## B3-A, T0, and Tsmall

`load_ipip300` reads the existing PAlign `Test-set.json` with its published `.xls` key, maps the exact 60 B1 short-form input item numbers to full-form numbers, and keeps the other 180 as targets. It namespaces local case IDs as `palign-ipip300:<case>` and converts invalid answers to missing before any calibration. The corpus is development-exposed; this loader does not make it an untouched external holdout. An IPIP-300 zero-target-label decoder and matched transfer evaluation still need to be implemented; the old target-adapted `transfer.py` does not satisfy that mode.

`t0_instruction_export` exports only source `role_definition` and `critic_formulations`, labelled as an **instruction-component** transfer using observed history. It drops NEO trait/facet formulations and cannot claim complete method transfer without a validated 30-facet mapping from permitted Twin inputs. `tsmall_schedule` generates exactly 12 configuration cells: 2 frozen models × GA/best-of-B × 3 seeds at B=40. It raises until both adapter data and readout probe gates are explicitly passed. A Tsmall search runner is **not** implemented by this module.

For B4, `m` is answers observed for the test respondent; `N_total` is all target-labelled humans consumed by training, tuning, and calibration. Rebuild the profile at each `m` from only those answers. With this adapter, target labels and exact repeats are absent from `history`; downstream subset generation must preserve that contract. Report per-endpoint observed denominators from the manifest, then split-specific eligible denominators after any endpoint exclusions. The adapter does not impute missing human labels or compute a savings claim.
