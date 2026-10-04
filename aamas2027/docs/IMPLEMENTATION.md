# HBS AAMAS comparison implementation

Spec: user authorizes all missing code and enqueueing the full evolutionary-baseline comparison on Euler. Research plan in RESEARCH_PLAN_20261004.md; include GA, DE, PromptBreeder, MAP-Elites, reflective GEPA, OPRO, ProTeGi, random, best-of-B. Implement official/native adapters where claiming their identities; never silently substitute a weaker implementation.

Isolation: /private/tmp/personbench-aamas-20261004, branch codex/hbs-aamas-comparison. Preserve original checkout and all existing results. Deployment goes into a new versioned Euler code directory with a new result root. Remote work, including setup/data/statistics, must use finite Slurm jobs. Short queue/file queries and rsync are allowed. No held-job release or broker mutation.

## Task 1: Correct scoring and deterministic objective

Own src/optim/scoring.py, src/optim/fitness.py and dedicated tests only. Correct bounded isotonic L2 projection (project before clipping), reject malformed simplex / missing targets consistently, crossfit Qbar from training fold, fixed fold partitions across candidates, genuine out-of-fold per-respondent scores for reflective selection, raw RPS objective support. Meaningful failing regression tests first. Preserve public legacy API where possible, add optional strict versioned behavior if necessary. Report exact changes, test command/results, limits. No remote work.

## Task 2: Optimizer adapters and exact budgets

Own arms/budget/readout adapter interfaces only after root approval. Preserve nine baseline families, add pinned official GEPA and native GigaEvo adapters if supported. Account mutation/hypermutation calls and actual fractional forward evaluations, eliminate phantom spending, ensure selected genotype written equals returned genotype. Seed mutator randomness explicitly. Test behavior, not just labels.

## Task 3: Scientific pipeline and analyses

Global immutable split excluding historical exposures, neutral seed, common train/opt/select/cal/eval, strong same-history decoder, proper respondent paired analysis, adjusted comparisons, uncertainty/censoring of m*, budget/savings analyses and transfer tracks with data readiness gates. New result root. No scope substitutions: incomplete adapters or datasets must be explicit and keep dependent jobs gated.

## Task 4: Deployment and verified enqueue

Pinned environment specification, finite CPU preflight/setup, GPU serving probes, full comparison arrays dependent on passed CPU/GPU gates, final evaluation/analysis dependency chain, durable job ledger. Verify actual Slurm job IDs, expanded matrix coverage, dependencies, resource requests, source hashes and no old output collisions. Queue submission is the requested terminal state; campaign completion is not required, but runnable correctness gates must be real.

## Rulings and interface scan

| Producer → consumer | Interface | Ruling |
|---|---|---|
| Task 1 → Task 2 | FitnessResult / per_respondent_scores | Keep existing fields, make selection scores honest; fixed folds and training-fold centering |
| Task 2 → Task 3 | Arm.run, ledger, selected genotype | New runner checks true cost and stores returned genotype independently of best-so-far |
| Task 3 → Task 4 | immutable manifest, results root | Queue never mutates manifests after submission; dependencies block invalid data/scorer paths |
| Research plan → implementation | frozen route versus full evolutionary comparison | Full fresh global comparison is mandatory; frozen reuse can support but cannot replace requested breadth |
| Skill examples → user infrastructure | screen / bare launches | User's Slurm-only rule overrides generic skill screen instructions |

New test data are fixtures only, never labeled scientific evidence. Public artifacts do not contain participant responses or credentials. No external messages or submission actions.
