# Euler + HSE: acceleration assessment, 5 October 2026

Status: **prepared allocation, not activated**. Existing Euler jobs and immutable code-v3 remain unchanged. HSE cannot yet run the campaign.

## Verified at 00:48–00:54 MSK

- Euler smoke 8692 still pending (Resources), scheduler estimate 04:02 MSK. Search 8693 and final 8695 wait for dependencies; no new scientific cell has started.
- HSE access works. `rocky`/`normal`, account `proj_1759`; current normal-QOS user cap is 20 GPUs / 180 CPUs, maximum 100 jobs. This is a cap, not a resource reservation.
- All 4 H100 and all 20 H200 GPUs on suitable HSE nodes are allocated. Actual release times can precede job walltime; future queue priority remains unknown.
- HSE has vLLM 0.28.0, but other package versions differ from the campaign environment; GEPA and GigaEvo are absent. Reuse the existing installation only as a base for isolated preparation; do not upgrade it in place.
- HSE default-cache directories have the exact pinned revisions, but contain only config.json (28–32 KiB each), **not model weights**. Euler copies occupy approximately 67 + 52 + 55 = 174 GiB. Matching directory names alone were an insufficient readiness signal.
- HSE corpus SHA-256 equals the source recorded in Euler's fixed panels: `7572de8b38af378e4740affe50705823efe085b1961e8ad19583d4e1cf1740c7`. Copy the existing panel manifest; never redraw panels on HSE.
- CPU audit 4376531 ran on cn-010 and failed on missing weight indices as intended. GPU witness 4376532 was canceled before execution because its prerequisite cannot succeed. No old HSE or Euler jobs were canceled.

## Prepared allocation profiles

`../hybrid-allocation-proposed.json` binds every cell index/ID to the existing manifest hash. Both profiles cover all 84 cells exactly once. All 14 method/objective variants within each model/seed block stay together, so cluster assignment does not systematically separate proposed methods from baselines.

| Profile | Euler / HSE GPUs | Euler / HSE cells | Search concurrency total |
|---|---|---|---|
| hse_2_gpu | 4 / 2 | 56 / 28 | 3 |
| hse_4_gpu | 4 / 4 | 42 / 42 | 4 |

For two HSE GPUs, seed 2 of both models goes to HSE; Euler keeps seeds 1/3. For four HSE GPUs, HSE gets Qwen seed 2 and Granite seeds 1/3; Euler gets the other blocks. Each site sees every method and both models. These assignments must be reconciled with live started cells before activation; never move an already running cell or duplicate it across sites.

## Conditional compute-time comparison

Assumptions solely for comparison: search+selection 4 h/cell, final 1 h/cell; all requested GPUs continuously available; no failures or queue gaps; both clusters ready together. These are **not measured durations**.

| Profile | Search | Final | Combined GPU stages |
|---|---:|---:|---:|
| Euler only | 42 waves × 4 h = 168 h | 21 waves × 1 h = 21 h | 189 h, about 7.9 days |
| Euler + HSE 2 GPU | 28 waves × 4 h = 112 h | 14 waves × 1 h = 14 h | 126 h, about 5.3 days |
| Euler + HSE 4 GPU | 21 waves × 4 h = 84 h | 11 waves × 1 h = 11 h | 95 h, about 4.0 days |

Add preparation, smoke, synchronization, CPU analysis and resource waits. Late HSE availability changes the best split. No evidence currently justifies promising a specific completion date.

## Activation work remaining

1. Stage exact weight revisions on HSE and verify all shards. A full model copy is about 174 GiB; do not replace the models or quantize just to fit another GPU.
2. Prepare isolated pinned main/native environments; validate actual GPU kernels and the two real-model all-arm smokes. Check a fixed development-only parity packet across sites; no final-evaluation access for tuning.
3. Make serving paths cluster-configurable, retaining exactly the same prompts, readout, objective and budget. Track site/runtime versions in receipts.
4. Activate one ownership manifest and establish verified result import to Euler. Slurm job IDs/dependencies are local to a cluster; a direct cross-cluster `afterok` dependency is not valid.
5. Replace only not-started affected Euler jobs and dependent final/analysis jobs once the hybrid route is runnable. Preserve attempt logs, corpus/panel hashes, cell identities and the global freeze gate.

Do not increase Euler's own share from four GPUs by taking resources from unrelated running projects. Do not change old holds or terminate valid experiments to meet a forecast.
