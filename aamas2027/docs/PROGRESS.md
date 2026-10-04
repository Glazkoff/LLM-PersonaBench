# SDD ledger — plan: aamas2027/docs/IMPLEMENTATION.md

2026-10-04: isolated checkout created at 32aab08. Original dirty documentation preserved. Skills: subagent-driven-development, test-driven-development, run-experiment. Read current code, curated infrastructure guidance and scientific plan.

Ruling: use separate Euler snapshot directory and output roots, not an in-place code update underneath held historical arrays. User authorized submitting this comparison, not changing other project jobs or their holds.

Task 1: in progress.
Task 2: pending.
Task 3: in progress (root architecture/data/runner).
Task 4: pending live readiness checks.

Task 1 complete; independent review found duplicate subset indices, fixed with a regression test.
Task 2 complete; official GEPA native fixture passed. Review findings fixed: strict JSON, native seed, full-panel shortlist, cost/failure reconciliation.
Task 3 core complete: global immutable panels, fixed seed, search/selection/final stages, strong statistical references and paired/censored analyses. External adapter separate work in progress.
Task 4 in progress: CPU exposure-v1 job 8668 completed, environment setup 8670 failed (nested venv did not inherit pandas), explicit dependency repair 8672 completed. No old jobs altered. Exposure-v2 extends exclusions to historical baseline redraws before any GPU search.
Ruling: use raw proper RPS as common native-additive search objective, retain calibrated-RPS and S0 objective contrasts; final calibrated RPS is fit only on cal. Costs and scientific meaning documented in README.
Ruling: use 2 isolated GPUs per search, 1 per final evaluation, at most 4 concurrent campaign GPUs; this avoids previous co-resident server OOM ambiguity.
Read-only whole-branch review: final GPU resource override and dependency graph validated; provenance mismatch checks added at freeze and analysis with regression tests. Native GigaEvo real-engine fixture passed locally (3 full evaluations, improved genotype selected, 8 native guided LLM calls). All services and generated data remain private.
External adapter now pins official Twin-2K-500 catalog/CSV inputs, checks download hashes, excludes exact/equivalent target QIDs, types ordinal versus nominal losses and refuses unreviewed endpoint selections. External model readout remains an explicit data-dependent gate, not an implied completed experiment.

Task 4 enqueue complete at 23:38 MSK: corrected code-v3 source manifest 8337ce817d69013f34b716a54cd1ff0c30cbb42ef46eed523b284c7ed7e8b9b8; CPU gate 8689 COMPLETED 0:0 (64 tests plus real native fixture); full arrays 8693/8695 accepted with 84 indices each, proper dependencies and 2-/1-GPU requests. Queue details and scientific limits are in QUEUE_STATUS_20261004.md. First failed packaging attempt 8675 and canceled new dependents 8676–8683 preserved as attempt 1; no old jobs modified.

Implementation worktree moved to the durable project sibling LLM-PersonaBench-aamas2027. External readout/Tsmall and literal savings are still separate incomplete tracks; they are not claims of the queued core comparison.
