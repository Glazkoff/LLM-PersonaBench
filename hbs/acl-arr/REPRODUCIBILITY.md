# Central-table and interval reproduction map

This is a route inventory checked on 12 November 2026, not a new computation or a claim of public end-to-end reproducibility. Previously passing audits were not repeated.

| Output | Existing finite path / evidence | Remaining boundary |
|---|---|---|
| Original R2 14 contrasts, 686 endpoint rows, intervals and Holm | `research-plan-20261102/receipts/20261110/r2-test/`: AUDIT.json, primary_contrasts.csv, uncertainty_contrasts.csv; job27183 completed, 4CPU/32GB/2h ceiling, observed71s | Restricted labels/predictions and frozen scheduler receipts needed |
| Main Table 1, controls figure/S1, CAL reselection intervals | Euler `/home/glazkov/hbs-a16-claude-controls-20261110/controls.py`, SPEC.md, SOURCE.sha256, RESULT.json; job27454, 4CPU/24GB/20min ceiling, observed11s | Script imports restricted audit_result; current public analysis/ is inspection source, not standalone |
| Answer curve/equal60 contrast | Euler `/home/glazkov/hbs-unified-20261103/run-20261106-v6/answers/` and `e3_export/`; audit27931, 2CPU/8GB/10min ceiling | Only m60 has both methods under equal answer count; no reconstruction of unsaved m<60 LLM forecasts |
| Main/S1 aggregate identities | Public `aggregates/audit.py`, controls.json, robustness.json | 149 identities, not independent confidence-interval replay |
| Historical main/S2/S4 | Preserved source/PDF and original frozen evidence | Unified portable replay entry point and complete dependency lock are not supplied by the public package; this gap remains open |

## Exact restricted R2 interface
Source: `research-plan-20261102/ops-20261106/r2-result-audit/audit_result.py` and adjacent `audit_uncertainty.py`; existing README is a historical pre-execution document, superseded in status by receipt27183. Do not edit that historical README into a claim it always described completion.

Required options: `--bundle`, `--lane`, `--gate-script`, `--score-gate`, `--out` (fresh directory), `--gate-script-sha256`, `--uncertainty-source-sha256`.
Euler root `/home/glazkov/hbs-unified-20261103`:
- bundle `bundle-20261106-stimulus-r2`
- lane `twin-20261106-stimulus-r2`
- gate `ops-20261106/postprobe-r2-review-2/verify_readout.py`
- successful receipt same directory `COMPARE_SCORE_VERIFIED.json`
- pinned gate SHA `ae38a2e7119709181c0b8b30e884fb180d6e9ae3da153fe0b265e4b4b610e49c`
- pinned auditor SHA `cc4497b569e9f2df5bb135c3deabdf4bbae826b28c01f5190b6e34c377f24f74`
- pinned uncertainty SHA `0bcc43d0d0d62c73d83f77e33a3e9f2099c28b4b68637299c5e804f007a8ee76`.

Use the existing `audit_result.sbatch`, with its documented absolute-path/hash environment variables, only in a new authorized replay allocation; never overwrite canonical outputs. A fresh independent numerical implementation already replayed central R2 intervals in27183. No reason to repeat now.

## Controls replay
Existing controls.sbatch sets `PYTHONPATH=/home/glazkov/hbs-unified-20261103/ops-20261106/audit-rehearsal-v1/ops-20261106/r2-result-audit`; interpreter `/home/glazkov/personbench-aamas-20261004/env/bin/python`; runs `controls.py /home/glazkov/hbs-unified-20261103` from a fresh copied source/SPEC directory after SHA checks. It reproduces 10000-draw conditional intervals and 2000 CAL/TEST reselecting draws. Actual existing files and imported audit module were checked present; no Slurm replay launched in this editorial stage.

## Packaging progress, second step on12 November
Implemented `manuscript-stage/replay/launcher.py`: public `cells` mode and restricted `r2`/`controls` plan-only modes, with explicit --execute inside a finite CPU allocation, new0700 output directories, source pins, no overwrite or auto-retry. Vendored auditor/uncertainty are identical to27183; controls uses its original7807404c…SPEC rather than the explanatory public SPEC. Dependency inventory and historical environment lock included.37 numeric cells are mapped and checked:14 Table1 cells,6 conditional stimulus-control cells,6 CAL-bootstrap bounds,3 equal60 cells and8 original R2 Granite estimate/interval/Holm cells. This is rounded-text agreement, not new inference; figure pixel geometry and all historical supplement cells remain unmapped.

Eight local tests exercise source/text tampering, absent/infinite/GPU allocation refusal, output collision and plan-only nonexecution. No real-data replay was run; previous successful audits remain evidence. The launcher retains the original restricted tree layout and cannot magically relocate absolute configuration paths. Remaining work: complete historical table/figure mapping, validate a separate relocation adapter if requested, and audit the anonymous release boundary. Serving drift and regeneration reproducibility remain separate from score replay.
