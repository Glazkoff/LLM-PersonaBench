# Resubmission plan: "Evolutionary Prompt Optimization for Personality-Driven Digital Twins"

NeurIPS 2026 submission #22622 (withdrawn 30 Jul 2026) → 2027 A\* venue.
Prepared 2026-09-25 from the OpenReview record (meta-review + 3 reviews), the submitted
source (`~/Downloads/NeurIPS2026___HBS___PromptEvo/main.tex`), the current ARR source in this
folder (`LLM-PersonaBench/paper/main.tex`, now a different paper), the project notes
(`ARR_OCTOBER_PLAN.md`, `E2_FINDING.md`, `PAPER_CHANGES.md`, `MODEL_REFRESH_PLAN.md`,
`HSE_RUN.md`, `STATUS.md`) and the codebase.

---

## 0. Executive summary: six decisions

1. **Change the objective before adding anything else.** Your own audit (`E2_FINDING.md`, ARR
   paper §3–4) proves that under the paper's fitness function, answer similarity
   `S = mean(1 − |x−y|/4)`, a per-item constant vector beats every evolved prompt in
   20/20 cells (by 13.6 pp raw, 5.8 pp after the orientation fix) and beats a *real human*
   from the same cluster. Reviewer FUHt asked for exactly this baseline. Any v2 that keeps `S`
   as the fitness dies at the first review. All new optimizer runs must optimise a strictly
   proper ordinal score (CRPS, i.e. `S_{1/2}`) and every table must show the constant floor
   and the human ceiling.
2. **Fix the orientation bug in the evolution pipeline before any new run.** 55 of 120
   IPIP-NEO-120 items are reverse-keyed in the corpus; `src/utils/personality_match.py` still
   scores raw model answers against recoded human answers (confirmed today: no
   reverse/keyed handling in `personality_match.py`, `person_type_opt.py`, `my_evaluator.py`).
   The fix exists only in `arr2026/scripts/_orientation.py`.
3. **Reframe the research question** from "our optimizer improves digital twins" to
   *"Which prompt optimizers, on which models, buy real personal information, and how much?"*
   This absorbs the advisor's request (many optimizers × many models) and makes both the
   positive and the negative outcome publishable. Measure the outcome in the unit your
   audit paper already defined: the answer-equivalent `m*` (real answers a persona is worth),
   plus CRPS, individuation and C2ST.
4. **Make the individual persona the primary unit; keep clusters as a secondary
   stratification.** The cluster-validity analysis you already ran (E1) finds no criterion
   supporting k=4 (silhouette 0.082, DB minimal at k=2, gap selects no interior k) though
   bootstrap ARI is stable. The prompt builder already supports per-respondent score
   modifiers ("individual regime"). Switching the primary unit dissolves objections A and E
   (k=4, "digital twin" overclaim) instead of defending them.
5. **Run a two-paper strategy.** Paper A (the audit, "What Is a Persona Worth?") goes to
   ARR 12 Oct 2026 → NAACL/COLING 2027, as already planned. Paper B (this plan) targets
   **ICML 2027 (abstract 16 Jan, paper 22 Jan 2027)** with ACL 2027 (ARR January cycle) and
   IJCAI 2027 (~1 Feb 2027) as parallel-deadline fallbacks and NeurIPS 2027 (21 May 2027) as
   the last resort. Paper B cites Paper A for the metric and prices its optimizers in `m*`.
6. **Pre-register the hypotheses (OSF) before the Tier-1 grid runs**, with equal LLM-call
   budgets per optimizer, ≥3 seeds per cell, 20 respondent splits, paired bootstrap CIs and
   Benjamini–Hochberg control across the grid. The NeurIPS paper reported single runs of a
   stochastic search on one 60/40 split of 100 users; the split-to-split SD of a *constant*
   baseline (1.3–2.5 pp) exceeds the reported +0.72 pp effect.

Everything below is the detail behind those six decisions.

---

## 1. Where things stand

| Fact | Value |
|---|---|
| Decision | Withdrawn by authors 30 Jul 2026 after meta-review (22 Jul). Ratings 3 / 1 / 3 (Borderline reject, Strong reject, Borderline reject). Quality 2/1/2, Clarity 2/1/2. |
| Meta-review (AC 6hcd) | Endorsed four issues: small/inconsistent gains; missing baselines; unjustified k=4 clustering (asks for seed stability, preprocessing sensitivity, alternative k); non-standard font. Called the problem "timely and important" and the method "reasonable and sound". |
| Reviewer 3bHJ | Per `ARR_OCTOBER_PLAN.md`, this review reproduces the NeurIPS canary phrases and is LLM-generated. Its points are independently made by FUHt and FHNv and endorsed by the AC, so nothing changes, but do not over-index on its Q2 "product/campaign" framing. |
| Submitted source | `~/Downloads/NeurIPS2026___HBS___PromptEvo/main.tex` (608 lines). Note: reviewers discuss a CARL/MAESTRO "chain-augmented" section (Type A/B, dynamic topology) that is **not** in this `main.tex`; recover the exact submitted source from Overleaf before reusing text. |
| Source in this folder | `LLM-PersonaBench/paper/main.tex` is **not** the NeurIPS paper. It is the ARR audit paper ("What Is a Persona Worth? Pricing LLM Human Simulation in Human Answers"), 8 pages ACL format, builds clean, 89 priced cells, 15 open models. |
| Method as implemented | EvoPrompt-style GA (`src/evolution/evoluter.py`, `GAEvoluter` only; DE is in the config schema but not implemented). Pop 4 × 5 generations, tournament selection, LLM crossover/mutation on a JSON genotype. GigaEvo is cited but not used as the engine. Fitness = answer similarity on 60 train users; test on 40. |
| Models used | GigaChat3-10B-A1.8B, GPT-4.1-mini, GPT-4.1-nano, Grok-4.1-fast, Qwen3-235B-A22B-2507 (configs under `configs/experiments/evoprompt_iter2/`). |
| Headline result | +0.72 pp mean answer similarity; MAE 35 +0.14 (worse), trait similarity −0.25, facet similarity −0.12. |
| What the audit already established | Proposition 1 (constant beats a faithful sampler under `S`; gap = GMD − MAD ≈ 7.75 pp); orientation defect (+0.09 → +0.63 per-item mean correlation after fix); k=4 has no criterion support but is bootstrap-stable; C2ST-AUC 0.994 vs human ceiling 0.564; VR 0.43; `m*` instrument with 89 cells; prompt search under `S_0` co-improves CRPS slightly (App. "reachability"). |
| Infrastructure ready | vLLM local provider with `likert_distribution()` logprob readout (`src/models/providers/local_vllm.py`); HF forward-pass readout (`h4_variance_ladder_hf.py`); Euler 8×H200 and HSE cHARISMa (V100/A100/H100/H200) Slurm recipes; cached weights for Qwen3.6-35B-A3B, Qwen3.8-27B, gemma-4-12B, GLM-4.7-Flash, gpt-oss-20b, granite-4.2-8b/30b, Apertus-8B, Qwen3-235B-FP8; inventories IPIP-NEO-120/300, IPIP-FFM-50 (BIG5), HEXACO, SD3 under `data/`; GigaEvo source at `~/Downloads/gigaevo-core-internal-main.zip`. |

---

## 2. Reviewer concern ledger (source-accounted)

Severity is the reviewer's own. Obligation: **must** = endorsed by the AC or raised by ≥2
reviewers; **should** = raised once with a concrete ask; **consider** = suggestion.
"Answer in v2" points to the experiment (§4) or presentation item (§5) that closes it.

| ID | Source | Concern (paraphrase) | Sev. | Oblig. | Cost scope | Answer in v2 |
|---|---|---|---|---|---|---|
| AC-1 | Meta-review | Empirical gains small and inconsistent; all reviewers agree | Major | must | re-analysis + new data | §3 reframe; E1–E4 (proper score, budget-matched optimizers, seeds, CIs) |
| AC-2 | Meta-review | Baseline comparison limitations | Major | must | new data | E3 (constant floor, human ceiling, random/paraphrase/best-of-N at equal budget) |
| AC-3 | Meta-review | k=4 unmotivated; wants seed stability, normalisation sensitivity, alternative k | Major | must | re-analysis | E9 (E1 results already computed; individual persona primary) |
| AC-4 | Meta-review | Font differs from template | Editorial | must | sentence | §5: stock template, no font changes |
| R1-1 | 3bHJ | Gains small/inconsistent; robustness not shown | Major | must | new data | E1, E4 |
| R1-2 | 3bHJ | Downstream (use-inspired) value not demonstrated; only IPIP matching | Major | must | new data | E6 (held-out items, other inventories, Twin-2K-500 behavioural block) |
| R1-3 | 3bHJ | Claims overreach: "reproduce human behavioural patterns", "digital twin" | Major | must | section | §3/§5: title and terminology; separate "questionnaire simulation" from "behaviour simulation" |
| R1-4 | 3bHJ | Fitness = answer similarity while trait/facet/MAE worsen; wants multi-objective or justification | Major | must | new data | Prop. 1 explains it; E1 proper-score fitness; E5 multi-objective Pareto |
| R1-5 | 3bHJ | Cluster validity (silhouette, seed stability, alternative k) | Major | must | re-analysis | E9 |
| R1-6 | 3bHJ | CARL/MAESTRO section disconnected; develop fully or move to appendix | Minor | should | section | §5: cut from Paper B (separate paper) |
| R1-Q4 | 3bHJ | Stability across evolutionary seeds | Major | must | new data | E4 (≥3 seeds/cell, 5 on headline cells) |
| R1-Q5 | 3bHJ | Baselines beyond hand-written prompt: paraphrasing, manual improvement, non-evolutionary rewriting, grid/random search | Major | must | new data | E3 |
| R1-L | 3bHJ | Limitations: questionnaire≠behaviour, demographic bias, self-report validity, synthetic-persona risks, ethics | Minor | should | section | §5 Limitations + Ethics |
| R2-0 | FUHt | Poor polish: 3-paragraph abstract/conclusion; Fig 1–2 whitespace; Table 1 tiny; Table 3 unreadable; inconsistent abbreviations/bold | Major (Clarity 1) | must | section | §5 |
| R2-1 | FUHt | Evidence does not support framing; <1 pp gain while MAE/trait/facet worsen | Major | must | re-analysis | §3, E1 |
| R2-2 | FUHt | Optimised and evaluated on the same IPIP-NEO-120; wants held-out questionnaires, behavioural tasks, free-form, decision scenarios | Major | must | new data | E6; disjoint conditioning/target item split in E1 |
| R2-3 | FUHt | Baselines: cluster item-wise majority, centroid-based prediction, numeric→verbal without evolution, random search at same budget, paraphrase without selection | Major | must | new data | E3 (all five included) |
| R2-4 | FUHt | Clustering/sampling: k, stability, demographic confounding; only 100 users/cluster, 60/40 | Major | must | re-analysis + new data | E9; E4 (≥200 held-out respondents/cluster, 20 splits) |
| R2-5 | FUHt | Statistics: CIs, significance, repeated splits, multiple-comparison correction | Major | must | re-analysis | E4 |
| R2-Q5 | FUHt | Sensitivity to the LLM used for mutation/crossover | Major | should | new data | E8 |
| R2-L | FUHt | No Limitations / Broader Impact section | Major | must | section | §5 (mandatory at every target venue) |
| R2-F | FUHt | Formatting: figure whitespace, Table 1 size, Table 3 unreadable | Editorial | must | section | §5 |
| R3-1 | FHNv | +0.72 pp while MAE/trait/facet worsen | Major | must | re-analysis | §3, E1 |
| R3-2 | FHNv | Vague definitions: clustering preprocessing, normalisation, k rationale, decoding parameters, number of runs, stochasticity handling | Major | must | section | §5 reproducibility block; E4 |
| R3-3 | FHNv | "Digital twin" overstated; cluster centroid is a coarse abstraction; same questionnaire family insufficient | Major | must | section + new data | §3, E6, E9 |
| R3-Q1 | FHNv | Baselines: per-cluster majority, empirical-distribution sampling, centroid-derived deterministic answers, random paraphrase search at same budget | Major | must | new data | E3 |
| R3-Q2 | FHNv | How was k=4 selected (silhouette, elbow, interpretability, convenience)? | Major | must | re-analysis | E9 (answer honestly: no criterion; report as design choice) |
| R3-Q3 | FHNv | Why "digital twin" rather than "cluster-level psychometric response simulator"? | Major | must | sentence | §5: adopt the reviewer's phrase |
| R3-F | FHNv | Template font modified | Editorial | must | sentence | §5 |
| POS-1 | AC, all | Timely problem; evolutionary procedure reasonable; reproducibility details useful; multi-model, multi-cluster evaluation; trait/facet metrics reported | Positive | — | — | Keep: pseudocode, prompts, code, compute reporting, multi-level metrics |

No comment is left unassigned. Contradictions: none between reviewers; FHNv rates
Significance/Originality 3 while FUHt rates 2/2, so the framing (not the idea) is what
moved scores.

---

## 3. The reframe: question, claims, hypotheses

### 3.1 Why the original claim cannot be repaired by more runs

Under `S`, the optimum is the per-item conditional median (Proposition 1 in the audit). A
search that succeeds under `S` is pulled toward a constant vector, which is exactly why
answer similarity rose while trait/facet profiles degraded: the optimizer was destroying the
response variance that carries the profile. Reviewers saw the symptom (R1-4, R2-1, R3-1) and
the paper filed it under future work. More optimizers and more models under the same
objective would reproduce the same symptom at larger scale.

### 3.2 The new research question

> Given a persona description derived from a person's psychometric profile, **can automatic
> prompt optimization increase the amount of person-specific information the LLM's answers
> carry**, measured as (i) a strictly proper ordinal score (CRPS) on held-out items and
> respondents, (ii) the answer-equivalent `m*`, and (iii) individuation/distinguishability
> (between-persona variance ratio, C2ST-AUC) — and **which optimizer families and which model
> generations do it?**

Working title options (avoid "digital twin"):

- *Can Prompt Optimization Buy Real Answers? Pricing Evolutionary and Reflective Prompt
  Optimizers for Persona Simulation*
- *Optimizing Personas Under a Proper Score: A Controlled Comparison of Prompt Optimizers
  Across Fifteen LLMs*

### 3.3 Pre-registered hypotheses (register on OSF before Tier-1 runs)

| ID | Hypothesis | Decisive measurement | If false |
|---|---|---|---|
| H1 | Under `S_0` (improper), optimizers improve `S_0` but do not improve CRPS on held-out items; under CRPS fitness they improve both. | Paired ΔCRPS, held-out respondents and disjoint target items, per (optimizer, model, cluster) | Report the direction measured; it is the central comparison either way |
| H2 | At equal LLM-call budget, selection-pressure methods (GA, GigaEvo, GEPA, OPRO) beat random search and paraphrase-only search on CRPS. | Δ vs random/paraphrase at budget B, BH-corrected | If random search ties, the paper's contribution becomes "budget-matched search suffices"; still a finding reviewers asked for |
| H3 | Reflective methods with textual feedback (GEPA, ProTeGi/TextGrad) reach a given CRPS with fewer evaluations than GA/DE. | Evaluations-to-threshold curves | Report curves |
| H4 | Optimized personas raise `m*` by a measurable amount on the IPIP-NEO-300 unseen-item family. | Δm* with paired bootstrap; cross-panel Spearman | If Δm* ≈ 0, the honest headline is "prompt optimization does not buy answers", with the mechanism (App. E of the audit) |
| H5 | Multi-objective search (CRPS + individuation) yields a non-degenerate Pareto front: individuation can be raised without losing CRPS. | Hypervolume vs single-objective; front plots | A degenerate front is itself a result about the objective landscape |
| H6 | Gains transfer across models less than across clusters: a prompt evolved on model X loses most of its gain on model Y. | Cross-model evaluation matrix | Either outcome informs deployment |

Design rule: **both outcomes of every hypothesis must produce a table the paper can print.**
This is what converts "gains are small" from a rejection reason into a finding.

ICML framing sentence (use it in the abstract and the first paragraph): *automatic prompt
optimizers inherit the pathologies of their objective; on a task whose standard metric has a
degenerate constant optimum, we show which optimizers exploit it, and how much of the gain
survives under a strictly proper score at equal budget.* That is an objective-misspecification
result about prompt optimization in general, with persona simulation as the case study, and
it is what makes the paper an ML-methodology paper rather than an application note. For the
ACL route, lead instead with the persona-simulation question and keep the same tables.

---

## 4. Experiment program

### P0. Prerequisites (week 1, blocking)

- **P0.1 Orientation fix in the pipeline.** Port `arr2026/scripts/_orientation.py` into
  `src/utils/personality_match.py` (or apply the recoding to the human side at load) and add a
  regression test that the per-item model/human mean correlation on the existing
  `evoprompt_iter2` artefacts moves from ≈0.09 to ≈0.63.
- **P0.2 Proper-score fitness.** Add `crps_fitness` (normalised ranked probability score /
  `S_{1/2}`) as the default `evolution.fitness`; keep `S_0` as an ablation arm. Two readouts:
  *belief readout* (logprobs over `{1..5}`, exact, open models) and *sampled readout* (k=5
  samples at T=1, API models). Report which readout each cell uses.
  **Primary fitness should be the *calibrated* CRPS**, i.e. the score of
  `Π_CDF[F₀ + α(Q − Q̄)]` with `α` fitted per candidate on the calibration panel (a 1-D
  search, negligible cost). Reason: the audit shows the raw score is dominated by prompt
  format and population location, and that raw personas lose to the cluster prior by
  0.036–0.095 CRPS in every cell; calibrated CRPS rewards exactly the person-specific
  deviation that `m*` prices. Raw CRPS and `S_0` are comparison arms, not the target.
- **P0.3 Disjoint conditioning/target items.** Build the persona from the input half of each
  scale (60 items) and score on the target half (60), as in the audit's App. "disjoint".
  Keep the IPIP-NEO-300 extra 180 items untouched for E6.
- **P0.4 Budget accounting.** Log evaluated candidates, prompt tokens, generated tokens and
  wall-clock per run; enforce an equal candidate-evaluation budget `B` across optimizers.
- **P0.5 Respondent protocol.** Per cluster: 40-respondent optimisation panel, 40-respondent
  calibration panel (hyperparameters, `α` for `m*`), ≥200-respondent frozen evaluation panel;
  20 random redraws for the split-noise floor.
- **P0.6 Pre-registration** of §3.3 on OSF with the analysis script committed.
- **P0.7 Recover the exact submitted NeurIPS source** (Overleaf) so the CARL/MAESTRO text and
  figures are available if needed for an appendix or a separate paper.

### E0. Pilot and budget calibration (week 2–3)

One model (Qwen3.6-35B-A3B, cached, fastest strong open model), one cluster, two optimizers
(GA, GEPA), B = 80 candidate evaluations, 3 seeds. Purpose: measure throughput under vLLM
prefix caching, verify P0.1–P0.4, and hit **Gate 1**: does the evolved persona improve
*calibrated* CRPS over the base persona on the frozen panel with a paired-bootstrap CI
excluding zero, and does its interpolated `m*` rise? Beating the raw constant floor is *not*
the gate: the audit shows every raw persona loses to the cluster prior by 0.036–0.095 CRPS,
so that bar would fail by construction and would be the wrong quantity anyway. Report
raw-vs-floor as a secondary column. If Gate 1 fails, the framing shifts to the H4-false
branch before the grid is spent.

### E1. Optimizer arms (the advisor's request)

Equal budget `B` candidate evaluations per run (pilot decides `B`; 80–120 is the target).
All operate on the same JSON genotype (role, trait formulations, facet formulations, critic)
so that differences are due to the search, not the representation.

| Arm | Family | Why it is in | Implementation |
|---|---|---|---|
| GA | EvoPrompt-style genetic (the NeurIPS method) | Anchor for continuity | existing `GAEvoluter` |
| DE | EvoPrompt differential evolution | Listed in the paper's own config; cheap second evolutionary arm | implement `DEEvoluter` (EvoPrompt template) |
| GigaEvo | MAP-Elites + multi-island + insight-aware mutation | The advisor's framework; quality-diversity is the natural fit for a Pareto/individuation objective | `gigaevo-core-internal-main.zip`; behaviour descriptors: prompt length, predicted dispersion (VR), trait-coverage |
| GEPA | Reflective prompt evolution with Pareto candidate pool (ICLR 2026 oral) | Current SOTA prompt optimizer; text feedback from per-item errors is available for free from the scorer | `gepa` open-source library; feedback string = worst items and their sign of error |
| OPRO | LLM-as-optimizer over scored trajectory | Classic non-evolutionary baseline reviewers name ("non-evolutionary LLM prompt rewriting") | ~150 lines: meta-prompt with top-k scored prompts |
| PromptBreeder | Self-referential co-evolution of mutation prompts | Second evolutionary SOTA; tests whether mutation-operator evolution matters | reimplementation (public ports exist) |
| ProTeGi or TextGrad | Textual-gradient beam search | Gradient-metaphor family; contrasts with population methods | TextGrad library or ProTeGi port |
| MIPROv2 (optional) | Bayesian instruction search (DSPy) | Only if time; GEPA already dominates it on public benchmarks | DSPy |

Fitness for the main grid: CRPS (belief readout). Ablation: the same GA under `S_0` on two
models, to demonstrate H1 directly. Mutator LLM fixed to one strong model across arms (E8
varies it).

### E2. Model slate

Continuity with the NeurIPS paper plus the 2026 generation the audit already has cached.
Open models get the exact belief readout; API models get the sampled readout.

| Tier | Models | Role |
|---|---|---|
| Open, core (Tier 1) | Qwen3-235B-A22B-2507 (original), GigaChat3-10B (original), Qwen3.6-35B-A3B, Qwen3.8-27B, gemma-4-12B-it, gpt-oss-20b | Full optimizer grid |
| Open, extension (Tier 2) | GLM-4.7-Flash, granite-4.2-30b, granite-4.2-8b, Apertus-8B, DeepSeek-V4-Flash (if the readout passes the ≥0.5-mass guard) | GA + GEPA + baselines only |
| API frontier (Tier 3) | GPT-4.1-mini (original), one current GPT-5-class model, one Claude model, one Gemini model | GA + GEPA + baselines, sampled readout, 1 seed; answers "do frontier models change the picture?" |

Report a release-date column (2024 → 2026) so model age becomes a time axis rather than a
weakness (`MODEL_REFRESH_PLAN.md`).

### E3. Baselines (every reviewer asked; zero or near-zero GPU cost)

Non-LLM floors and ceilings, computed on the frozen evaluation panel with the same score:

1. per-item cluster mean (constant), 2. per-item cluster majority, 3. global mean,
4. empirical-distribution sample, 5. centroid-derived deterministic answers (facet score →
Likert), 6. **real same-cluster human** (ceiling), 7. the audit's conditional model `D_m` at
m = 0, 5, 20, 60 answers (this is what `m*` is priced against).

LLM baselines at the same budget `B`: 8. hand-written base prompt (the paper's "before"),
9. one expert-improved prompt, 10. random prompt search (B samples, best-on-train),
11. paraphrase-only search (B paraphrases, no selection pressure), 12. best-of-B initial
population (selection without evolution), 13. **wrong-persona control** (the evolved prompt
with respondents' personas permuted within the panel; the audit recommends this as a
standing control and no persona study reports it). Items 10–12 separate "search helps" from
"evolution helps"; item 13 separates "the persona carries the person" from "the prompt
carries the population".

### E4. Seeds, splits and statistics

- ≥3 evolutionary seeds per (optimizer × model × cluster) in Tier 1; 5 seeds on the headline
  cells (best optimizer × 2 models).
- 20 respondent redraws for every deterministic baseline (split-noise floor).
- Paired bootstrap (B = 2000) over evaluation respondents for every Δ; BH at q < 0.05 across
  the whole grid; report effect sizes in CRPS points and in `m*` answers, not only p-values.
- Report the number of cells where the optimizer beats (a) its own base prompt, (b) random
  search, (c) the constant floor; all three counts in one table.

### E5. Multi-objective search (the new methodological contribution)

Objectives: CRPS (accuracy under a proper score), between-persona individuation
(`VR_between` or aligned amplitude `A`), and distinguishability (C2ST-AUC, style-only rung
reported alongside). Two arms: NSGA-II over the GA genotype, and GigaEvo's MAP-Elites with
these as descriptors; GEPA's Pareto pool as a third. Deliverable: the Pareto-front figure per
model, hypervolume vs single-objective runs, and the "corner" prompts. This directly answers
R1-4 and gives the paper a figure to be built around.

### E6. Transfer and held-out evaluation (answers R1-2, R2-2, R3-3)

1. Held-out respondents (always).
2. Held-out items within IPIP-NEO-120 (disjoint conditioning/target halves, P0.3).
3. **Unseen item family**: the 180 IPIP-NEO-300 items outside the 120, same respondents (the
   audit's `h24_ipip300_readout.py` panel of 300 respondents).
4. **Other inventories** with the frozen evolved prompt: IPIP-FFM-50, HEXACO, SD3 (all under
   `data/PAlign`), keyed with published item keys (HEXACO caveat as in the audit).
5. **Behavioural block**: Twin-2K-500 (2,058 people, 500 questions incl. Big Five,
   behavioural-economics replications and a pricing survey; public on Hugging Face). Condition
   the persona on the Big Five block, predict held-out behavioural/pricing answers. This is
   the "use-inspired" validation every reviewer asked for, at zero data-collection cost.
   Report it as an exploratory, pre-registered secondary outcome.

### E7. Price the optimized personas in `m*`

Run the audit's answer-equivalent protocol (`h26_answer_equivalent.py`) on base vs evolved
prompts for every Tier-1 cell. The headline figure of Paper B is *Δm\* per optimizer per
model*, with the constant floor at m = 0 and the ladder from Paper A as context. This is what
links the two papers and makes the unit of the claim "answers", not percentage points.
Use the **interpolated** `m*` and the paired loss gap as the effect size; the integer crossing
is too coarse to detect the sub-answer shifts an optimizer will most likely produce (the
audit's top cell moved by fractions of an answer between panels). Report cross-panel
Spearman of `Δm*` as the replication check, as Paper A does for `m*` itself.

### E8. Mutator-LLM sensitivity and cross-model transfer

- Mutator ∈ {same model as evaluated, one fixed strong model, one small model} for GA and
  GEPA on two evaluated models (R2-Q5).
- Cross-model matrix: evolve on X, evaluate frozen on Y for all Tier-1 pairs (H6).

### E9. Clustering and the unit of analysis

- Reuse E1 (`e1_cluster_validity.py`, `h10_stability_null.py`): silhouette/CH/DB/gap over
  k = 2..15 under three preprocessings, bootstrap ARI, surrogate-null calibration, demographic
  confound (Cramér's V, permutation) — the AC's list, verbatim.
- State plainly: no internal criterion selects k = 4; k = 4 is the largest bootstrap-stable
  k and is kept as a stratification for comparability with prior work. The primary unit is
  the individual persona (per-respondent score modifiers), which is what the released prompt
  builder already does.
- Add one ablation arm: a **single global template** (no cluster-specific wording, individual
  modifiers only) evolved once, against the four cluster-specific templates. If the global
  template matches them, the cluster layer is shown to be unnecessary rather than merely
  unjustified, and objection A closes for good.
- Rename throughout: "cluster-level psychometric response simulation" / "persona simulation";
  never "digital twin" as a claim.

### E10. CARL/MAESTRO chain evolution

Recommendation: **cut from Paper B.** Reviewer 3bHJ called it disconnected; at 8–9 pages it
cannot be developed. If the advisor wants chain-topology evolution in, run it as one more
GigaEvo arm under the same objective and budget (so it is comparable), otherwise keep it for
the CARL/GigaEvo paper.

### Compute budget (to be calibrated by E0)

Assumptions: belief readout, 40 optimisation respondents × 60 target items per candidate,
persona prefix cached under vLLM, B = 80.

| Tier | Runs | Forward passes | Rough GPU-hours |
|---|---|---|---|
| Tier 1: 7 optimizers × 6 models × 4 clusters × 3 seeds | 504 | ≈ 97 M | 50–250 (depends on tokens/s; 30B-class ≈ 8–40 h per model) |
| Tier 2: 3 arms × 5 models × 4 clusters × 3 seeds | 180 | ≈ 35 M | 20–100 |
| E5 multi-objective: 3 arms × 4 models × 4 clusters × 3 seeds | 144 | ≈ 28 M | 15–80 |
| E6–E8 frozen evaluations | — | ≈ 10 M | < 20 |
| Tier 3 API: 3 arms × 4 models × 4 clusters × 1 seed, k = 5 samples | 48 | ≈ 0.6 M calls (whole-questionnaire calls) | budget line, not GPU |

All of Tier 1 fits on Euler's 8×H200 in two to three weeks with HSE as overflow. The
bottleneck is engineering (P0 + optimizer adapters), not GPU.

---

## 5. Presentation plan (worth a full rating point on its own)

- **Template discipline**: stock ICML/ACL style, no font or spacing changes (AC-4, R2-F,
  R3-F). Generate every table from result files (`make_paper_assets.py` pattern) so no
  transcription errors survive.
- **Abstract**: one paragraph, numbers-first: question, design (optimizers × models ×
  clusters × seeds at equal budget), headline ΔCRPS/Δm\* with CI, the transfer result, the
  negative-or-positive multi-objective result, release statement.
- **Figure 1**: Δm\* (or ΔCRPS) per optimizer per model, CIs, constant floor and human ceiling
  as horizontal lines. **Figure 2**: Pareto fronts (CRPS vs individuation). **Figure 3**:
  evaluations-to-threshold curves. **Figure 4**: cross-model transfer heatmap.
- **Tables**: one summary table in the body (optimizer × metric, mean over models with CI,
  wins vs floor / vs random / vs base); the full grid in the appendix. Table 3's 20 × 11 delta
  grid (called "completely unreadable") must not return in that form.
- **Method section**: state decoding parameters, readout, seeds, splits, preprocessing,
  normalisation, k rationale, mutator model, budget definition (R3-2).
- **Mandatory sections**: Limitations (questionnaire ≠ behaviour; corpus skew 18–25,
  US/UK/CA; self-report validity; cluster-level ≠ individual; readout ≠ deployment; API vs
  belief readout comparability), Ethics/Broader Impact (synthetic personas must not replace
  participants; stereotype risk), Reproducibility (anonymised repo, per-cell results, seeds,
  budgets).
- **Terminology**: adopt FHNv's phrase; use "persona" for the prompt and "respondent
  simulator" for the system.
- **Anonymity**: no author-identifying repo links, no reference to the NeurIPS round, third
  person for the audit paper ("a concurrent audit shows…", cited as an anonymous preprint).

---

## 6. Venue analysis: top-5 nearest A\* venues

All ranks are CORE (ICORE) A\* unless noted. Dates verified 2026-09-25 against the venue
sites or their CFPs; where a 2027 CFP is not yet posted the date is marked *projected*.

| # | Venue | Deadline | Fit evidence | Feasibility for Paper B | Risk |
|---|---|---|---|---|---|
| 1 | **ICML 2027** (South America) | Abstract **16 Jan 2027**, paper **22 Jan 2027** AoE | Prompt-optimization methods papers live here (PromptBreeder, ICML 2024; LLM-as-optimizer line). Controlled optimizer × model comparison with proper-score evaluation is squarely ML methodology. | **Primary target.** 17 weeks; Tier 1 + E5–E7 fit. | ICML reviewers will want the optimizer comparison to be the contribution, not the persona application; make E1/E5 the spine. |
| 2 | **ACL 2027** via ARR January 2027 cycle | ARR January 2027 (exact day TBA; ACL commitment TBA) | The persona-simulation literature is an ACL literature: Hu & Collier (ACL 2024), Cao et al. (NAACL 2025, survey-distribution simulation), Li et al. (Findings ACL 2025, digital-twin benchmark), EMNLP 2025 persona user-simulation. Reviewers know the constant-floor/human-ceiling argument. | **Parallel-deadline fallback**; same manuscript in ACL format (8 pages + unlimited appendix). | New ARR sustainable-reviewing rule from Oct 2026: a qualified author must serve as reviewer or the paper is not guaranteed review. Dual submission with ICML forbidden: choose one. |
| 3 | **IJCAI 2027** (Kyoto, 7–13 Aug 2027) | ~**1 Feb 2027** *(projected; CFP not yet posted)* | "AI and social sciences" and agents tracks; IJCAI-ECAI 2026 hosted value-persona work (VALE workshop). Broad AI audience receptive to use-inspired framing. | Fallback if ICML slips by one to two weeks. | Shorter page budget (7 + refs); stricter on novelty of method. |
| 4 | **NeurIPS 2027** (Europe) | Abstract **14 May 2027**, paper **21 May 2027** AoE | The original venue; AC 6hcd called the problem timely and the method sound. Returning with all four AC-endorsed issues closed is the standard path. Datasets & Benchmarks track is an option if the protocol (floors, ceilings, `m*`) becomes the contribution. | Last resort; allows Tier 2/3 and Twin-2K-500 in full. | Same reviewer pool may recognise the paper; the bar for "gains are small" will be applied to the new numbers. Six extra months of competition in a hot area. |
| 5 | **The Web Conference (WWW) 2027** (Dublin, 10–14 May 2027) | Abstract **11 Oct 2026**, paper **18 Oct 2026** AoE | Web/social-computing track takes synthetic-respondent work (e.g., "Assessing the Reliability of Persona-Conditioned LLMs as Synthetic Survey Respondents", WWW 2026 companion). | **Not feasible for Paper B** (3 weeks). Feasible only as an alternative home for **Paper A** instead of ARR Oct 12 (pick one; no dual submission). | Fit is weaker than ARR for the audit; only worth it if the team prefers an A\* label over NAACL (CORE A). |

Also considered and rejected for Paper B: **ICLR 2027** (paper deadline is today, 25 Sep
2026); **AAMAS 2027** (A\*, paper 8 Oct 2026, too soon); **AAAI 2027** (closed July 2026);
**KDD 2027** cycle 2 (A\*, Feb 2027, weak topical fit); **COLM 2027** (~late March 2027
*projected*; not CORE-ranked but the best topical fit outside ACL/ICML, a good secondary
fallback); **EMNLP 2027** via ARR (~May 2027, A\*, natural fallback after IJCAI);
**NAACL 2027** via ARR Oct 12 (CORE A, not A\*; where Paper A is going);
**CSCW 2027** (A, Oct 15, HCI framing not what the paper is).

Recommendation: submit Paper B to **ICML 2027**; if Gate 2 (§7) is missed, roll to **ARR
January 2027 → ACL 2027** without changing content; IJCAI 2027 as the third option; NeurIPS
2027 if the Twin-2K-500 behavioural block turns out strong enough to be the headline.

---

## 7. Timeline to ICML 2027 (17 weeks) with gates

| Window | Work | Gate |
|---|---|---|
| Sep 25 – Oct 3 | P0.1–P0.7. Freeze design, pre-register. Recover submitted source. (Paper A team finishes ARR Oct 12 in parallel.) | Orientation regression test passes; CRPS fitness unit-tested against `arr2026/scripts/_scoring.py` |
| Oct 4 – Oct 17 | Optimizer adapters: DE, GigaEvo, GEPA, OPRO, PromptBreeder, ProTeGi/TextGrad; baselines 8–12; budget accounting. E0 pilot on Qwen3.6 × cluster 0. | **Gate 1** (Oct 17): ≥1 optimizer beats the constant floor under CRPS on the frozen panel with CI excluding zero. If not, switch to the H4-false framing now. |
| Oct 18 – Nov 14 | Tier-1 grid on Euler (HSE overflow). E3 baselines and 20-split noise floor (CPU). | Weekly seed-consistency check; kill arms that fail the readout guard |
| Nov 15 – Nov 28 | E5 multi-objective; E8 mutator sensitivity; Tier 2; Tier 3 API runs. E6 frozen transfers (IPIP-300, FFM-50, HEXACO, SD3, Twin-2K-500). | **Gate 2** (Nov 28): all Tier-1 cells have ≥3 seeds; E6 transfers computed. If < 80 % of cells done, target ARR January instead of ICML. |
| Nov 29 – Dec 19 | E7 pricing in `m*`; statistics (bootstrap, BH); figures 1–4; internal review: ARS 5-seat panel + one Codex `gpt-6-sol` pass (medium reasoning; note: on 2026-09-25 both the Codex MCP and `codex exec` rejected `gpt-6-sol` under a ChatGPT-account login, so this needs an API-key Codex login or a different reviewer model). | Result freeze Dec 19 |
| Dec 20 – Jan 9 | Full draft in ICML template; Limitations, Ethics, Reproducibility; anonymised repo and per-cell results bundle; claim audit against result files. | Draft complete Jan 9 |
| Jan 10 – Jan 22 | Second internal review, kill-argument pass, polish. Abstract Jan 16. Submit Jan 22. | Submitted |

---

## 8. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| No optimizer beats the constant floor under CRPS (Gate 1 fails) | Medium. The audit's reachability appendix shows prompt search under `S_0` co-improves CRPS only slightly. | Pre-registered H4-false branch: the paper becomes "prompt optimization does not buy real answers; here is the budget-matched evidence and the mechanism". Publishable at ACL/EMNLP; harder at ICML. Decide at Gate 1, not in December. |
| Reviewers read Paper B as incremental to Paper A | Medium | Make the optimizer comparison, equal-budget protocol and multi-objective fronts the contribution; Paper A supplies the unit. Never restate Paper A's results as new. |
| Readout failures on 2026 reasoning-first models (mass on answer tokens < 0.5) | High for GLM-5.x, DeepSeek-V4, Qwen3.8-Next per the audit | Keep those in Tier 2 only; report failures; use the prefill/raw-completions route the audit documents. |
| API vs belief readout not comparable | Certain | Report Tier 3 separately with the sampled readout; do not pool. |
| GigaEvo integration cost | Medium | Start the adapter in week 2; if it slips, GigaEvo runs on two models only and is reported as such. |
| Twin-2K-500 behavioural block shows nothing | Medium | It is a pre-registered secondary outcome; a null is reported as a null. |
| ARR reviewer-service requirement (from Oct 2026) | Certain if ACL route is used | Nominate a qualified co-author as ARR reviewer at submission. |
| Version drift between the submitted PDF and the recovered `main.tex` | Confirmed | P0.7. |

---

## 9. How the new paper answers each reviewer (cover-letter skeleton, if resubmitting to NeurIPS 2027)

- *Gains small and inconsistent* (AC-1, R1-1, R2-1, R3-1): we show why under the original
  metric a constant is optimal (Prop. 1), replace the objective with a proper score, and
  report budget-matched optimizers with ≥3 seeds, 20 splits, paired bootstrap CIs and BH
  control; effects are quoted in CRPS points and in real answers (`m*`).
- *Baselines* (AC-2, R1-Q5, R2-3, R3-Q1): twelve baselines including constant floor, human
  ceiling, random and paraphrase search at equal budget, best-of-B, and the conditional
  answer model.
- *Clustering* (AC-3, R1-5, R2-4, R3-Q2): full validity analysis (k = 2..15, three
  preprocessings, bootstrap ARI, surrogate nulls, demographic confounds); primary unit moved
  to the individual persona.
- *Evaluation on the optimised instrument only* (R1-2, R2-2, R3-3): disjoint item halves,
  180 unseen IPIP-NEO-300 items, three further inventories, Twin-2K-500 behavioural block.
- *Fitness vs profile* (R1-4): proper-score fitness plus multi-objective Pareto fronts.
- *Overclaim* (R1-3, R3-3, R3-Q3): retitled; "persona simulation"; no "digital twin" claim.
- *Mutator sensitivity* (R2-Q5): E8.
- *Presentation* (AC-4, R2-0, R2-F, R3-F, R3-2): stock template, generated tables,
  single-paragraph abstract, complete protocol block, Limitations and Ethics.
- *CARL/MAESTRO* (R1-6): removed.

---

## 10. Immediate next actions (this week)

1. Confirm the target (ICML 2027 primary) and the two-paper split with the advisor and the
   co-author list for Paper B.
2. P0.1: port the orientation transform into `personality_match.py`; add the regression test.
3. P0.2: implement `crps_fitness` on the belief readout in `my_evaluator.py`; unit-test
   against `arr2026/scripts/_scoring.py`.
4. Recover the exact submitted NeurIPS source from Overleaf (P0.7).
5. Write the OSF pre-registration from §3.3 (one page).
6. Start the GigaEvo and GEPA adapters (highest integration risk).
7. Book Euler/HSE allocations for Oct 18 – Nov 28.

---

## Sources checked for §6

- ICLR 2027 CFP (abstract 18 Sep, paper 25 Sep 2026): https://iclr.cc/Conferences/2027/CallForPapers
- ARR dates (Oct 12 2026 cycle → NAACL/COLING 2027, commitment 23 Dec 2026; January 2027 cycle → ACL 2027): https://aclrollingreview.org/dates ; NAACL 2027 CFP: https://2027.naacl.org/calls/main_conference_papers/ ; ARR sustainable reviewing policy: https://aclrollingreview.org/sustainable-reviewing-2026
- ICML 2027 (abstract 16 Jan, paper 22 Jan 2027): https://icml2027.eurac.edu/en/call-for-papers
- IJCAI 2027 (Kyoto 7–13 Aug 2027; deadline projected ~1 Feb 2027): https://www.ijcai.org/future_conferences
- NeurIPS 2027 (abstract 14 May, paper 21 May 2027, projected from tracker): https://researchtheta.com/conferences/neurips-2027/
- WWW 2027 (abstract 11 Oct, paper 18 Oct 2026): https://www2027.thewebconf.org/important-dates/
- AAMAS 2027 (abstract 1 Oct, paper 8 Oct 2026): https://warwick.ac.uk/fac/sci/dcs/aamas2027/calls/
- AAAI-27 (closed 28 Jul 2026): https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/
- KDD 2027 research track cycles: https://kdd2027.kdd.org/research-track-call-for-papers/
- COLM 2027 (Oct 6–9 2027; deadline projected late March): https://mlciv.com/ai-deadlines/conference/?id=colm27
- CORE ranks: https://portal.core.edu.au/conf-ranks/
- GEPA (ICLR 2026 oral): https://arxiv.org/abs/2507.19457 ; GigaEvo: https://arxiv.org/abs/2511.17592
- Twin-2K-500 dataset: https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500 ; Personality Alignment / PAPI: https://arxiv.org/abs/2408.11779
- Multi-objective prompt optimization precedents: CAPO https://arxiv.org/abs/2504.16005 ; CRAFT https://arxiv.org/abs/2606.04661
