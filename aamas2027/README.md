# PersonBench AAMAS 2027 comparison

This package implements a new global-panel comparison. It does not reuse old
cluster-conditioned scores as fresh evidence and does not claim scientific
superiority before the campaign finishes.

Current deployment and verified queue IDs: [Euler status](docs/QUEUE_STATUS_20261004.md).

The immutable matrix has **84 searches**: GA, DE, PromptBreeder, MAP-Elites-style,
GEPA-style, official GEPA 0.1.4, native GigaEvo internal 2.20.1, OPRO, ProTeGi,
random and best-of-B on Qwen3.6-35B-A3B and Granite4.2-30B with three optimizer
seeds (66), plus GA/calibrated-RPS, GA/S0 and MAP-Elites/calibrated-RPS controls
(18). The common main search objective is raw normalized ordinal RPS, which
supports additive native GEPA feedback; final calibrated RPS is fitted only on
calibration people. This change from the earlier narrower planning grid is
intentional: the user requires the full evolutionary comparison.

All methods share a neutral 5-trait/30-facet schema, the same split and model
snapshots, and a Qwen3.8-27B mutator. No target-derived cluster chooses a persona.
The train/opt/select/cal/eval counts are 8000/80/120/120/600. Historical explicit
IDs, saved response fingerprints, queued panels, headline baseline panels and
split-noise redraws are excluded; coverage and reconstruction limitations are
recorded in the exposure receipt. This is a newly reserved panel within a
previously studied public corpus, not a claim that the whole corpus was never
seen during the research program. IPIP300 is also previously exposed.

The search cap is 80 × 80 × 60 = **384,000 executed forecasts per cell**.
Shortlist selection has a separate maximum of 5 × 120 × 60 = 36,000 forecasts.
Legitimate fractional underuse is reported; no empty ledger charges fill it.
Every HTTP forecast is reconciled, implicit retries are disabled, and unknown
server-side cost makes a cell fail. Parse failures use zero forecasts and have
bounded attempts. Token counts and total native guided-pipeline LLM requests
are separate from forecast counts. GigaEvo's LLM counter includes auxiliary
insight calls and is not a count of just mutations.

No final evaluation job can start until all selected JSON personas and the
strongest comparator per model are frozen by hash. The common shortlist
selector sees only full optimization-panel candidates; native minibatch scores
cannot win a shortlist slot. Final files retain calibration and evaluation
probabilities, masses, the exact selected identity, a deranged-persona control,
and complete denominators. Missing readouts fail a cell rather than deleting
people. Main analysis uses respondent pairing, seed/respondent bootstrap,
respondent sign flips and Holm corrections. It compares calibrated simulator
losses to raw proper-RPS statistical references; those are both losses on the
same predictions/tasks, with their different calibration explicitly stated.

The statistical references are training-only empirical prior, ridge ordinal
CDF and ExtraTrees on the same observed 60 answers, selected on the separate
selection panel. Retained-information curves use three frozen nested input
orders. m* is censored beyond the measured grid and **is not literal saved
answers**. External behavior, actual questionnaire savings and cross-model
transfer remain distinct tracks; they cannot be inferred from this main table.

## Deployment

Use `slurm/submit.sh CAMPAIGN CODE DATA_JOB ENV_JOB` as the authoritative entry.
It supplies the one-GPU override for final evaluation. Direct use of gpu.sbatch
without that override for `final` is rejected. Search requests two GPUs (one
readout, one mutator), array concurrency two; final evaluation requests one GPU,
concurrency four. All services bind to 127.0.0.1 and die with the allocation.

The chain is CPU tests/native fixture -> two real-model all-arm smokes -> 84
searches -> frozen selection -> 84 final evaluations -> paired analysis, with
CPU decoder stages in the appropriate branches. Every stage has finite
walltime. Queue submission is not evidence of completion or a positive result.
A failed gate blocks dependent jobs. Preserve logs and results when recovering;
do not rerun test-driven selection after viewing evaluation outcomes.

Native GigaEvo is a pinned private dependency, not the public release. Its native
engine, guided pipeline and archive run with a JSON-format prompt adaptation,
shared validator and external budget stopper. Source SHA and exact invocation
are recorded. Release/access to that dependency is a paper reproducibility
requirement, not something this private cluster run proves.
