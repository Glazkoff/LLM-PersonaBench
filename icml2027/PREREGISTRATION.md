# Pre-registration: prompt optimizers for persona simulation under a proper score

Status: **DRAFT — freeze before the first Tier-1 job is submitted.** Freezing means: commit
this file, record the commit hash below, and register the same text on OSF. After freezing,
changes go in the "Amendments" section with a date and a reason; the original text stays.

Frozen at commit: `<hash>` · OSF: `<url>` · Date: `<YYYY-MM-DD>`

## 1. Question

Given a persona description derived from a respondent's psychometric profile, can automatic
prompt optimization increase the person-specific information an LLM's questionnaire answers
carry, and which optimizer families and model generations do so?

## 2. Design summary

- **Unit of analysis**: a cell = (optimizer arm, model, cluster, seed). Clusters are the four
  k-means clusters of the released corpus, used as a stratification only; the persona is
  individual (per-respondent facet scores from the input-half items).
- **Budget**: every arm gets exactly **B = 80 candidate evaluations** per cell. One candidate
  evaluation = one genotype scored on the full optimisation panel. Token counts are logged and
  reported but do not define parity.
- **Panels** (per cluster, drawn once per seed with `numpy.random.default_rng(seed)` from
  respondents not used by any prior run of this project): optimisation n=40, calibration n=40,
  evaluation n=200 (frozen). Baselines are additionally redrawn 20 times to estimate the
  split-noise floor.
- **Items**: within each of the 30 IPIP-NEO facets, the first two items in item-id order are
  *input* (persona construction), the last two are *targets* (scored). 60/60. Unseen-item
  transfer uses the 180 IPIP-NEO-300 items outside the short form (`data/PAlign/Test-set.json`).
- **Readout**: belief readout (logprobs over the five answer tokens, thinking disabled,
  assistant turn prefilled) for open models; sampled readout (k=5 at T=1) for API models,
  reported separately. A cell whose mean mass on the answer tokens is below 0.5 is excluded
  and listed.
- **Orientation**: model simplices are reversed on the corpus's reverse-keyed items before any
  score is computed (`src/optim/orientation.py`).
- **Seeds**: {1, 2, 3} for every Tier-1 cell; {1, 2, 3, 4, 5} for the headline cells (best arm
  × two models), chosen after Tier-1 by the pre-specified rule "largest mean ΔRPS_cal".

## 3. Outcomes

Primary:
- **RPS_cal**: normalised ranked probability score (`_scoring.rps`, lower is better) of the
  *calibrated* forecast `Π_CDF[F₀ + α(Q − Q̄)]` on the evaluation panel, with `α` from the grid
  in `configs/panels.yaml` selected on the calibration panel. During search, the fitness is the
  same quantity on the optimisation panel with `α` chosen by 2-fold cross-fitting inside that
  panel (halves A/B: select on A, score B, and vice versa; average). The calibration and
  evaluation panels are never read during search. Effect size: ΔRPS_cal = evolved − base.
- **m\*_interp**: interpolated answer-equivalent of the evolved vs the base persona
  (protocol of `arr2026/scripts/hyp/h26_answer_equivalent.py`, budgets
  {0,1,2,3,5,8,12,20,30,45,60}, ridge retuned at every budget on the calibration panel).

Secondary: RPS_raw, S₀ (the audited metric), α\*, VR_between (individuation), C2ST-AUC with
its style-only rung, transfer RPS_cal on IPIP-NEO-300 targets, cross-model transfer matrix,
evaluations-to-threshold curves.

## 4. Hypotheses and decision rules

| ID | Hypothesis | Test | Decision |
|---|---|---|---|
| H1 | Under `S₀` fitness, arms improve `S₀` but not RPS_cal; under RPS_cal fitness they improve both. | GA under `S₀` vs GA under RPS_cal on two models × 4 clusters × 3 seeds; paired bootstrap on ΔRPS_cal | Supported if the `S₀`-fitness ΔRPS_cal CI includes 0 or is positive (worse) while the RPS_cal-fitness CI is below 0 in ≥ 6/8 (model, cluster) pairs |
| H2 | At equal budget, selection-pressure arms beat random search and paraphrase-only search. | Δ vs random and vs paraphrase per cell, BH q<0.05 over the grid | Supported if ≥ 50% of cells for an arm are significant in the right direction after BH |
| H3 | Reflective arms (GEPA, ProTeGi) reach a target RPS_cal with fewer evaluations than GA/DE. | Evaluations-to-threshold (threshold = base − 0.005) per cell; Wilcoxon over cells | Supported if median evaluations are lower with p<0.05 |
| H4 | Optimized personas raise `m*` on the IPIP-NEO-300 unseen items. | Δm\*_interp with paired bootstrap; cross-panel Spearman of Δm\* | Supported if pooled Δm\* CI excludes 0 and cross-panel ρ>0.5 |
| H5 | Multi-objective search yields a non-degenerate front: VR_between rises without RPS_cal loss. | Hypervolume vs single-objective; front dominance count | Supported if the front contains ≥ 1 point dominating the single-objective best on both axes in ≥ 50% of cells |
| H6 | Evolved prompts transfer worse across models than across clusters. | Fraction of ΔRPS_cal retained, cross-model vs cross-cluster; paired bootstrap | Supported if retention_model < retention_cluster with CI excluding 0 |

All six are reported whichever way they come out.

## 5. Statistics

Paired bootstrap over evaluation respondents, B = 2000 resamples, seed 20260925, percentile
95% intervals. Benjamini–Hochberg at q < 0.05 across the full set of primary tests (one per
cell per comparison). Effect sizes in RPS points and in answers (`m*`). Cells are never pooled
across readout types.

## 6. Exclusions, fixed in advance

- Mass on answer tokens < 0.5 (mean over the panel) → cell excluded, listed in an appendix.
- A candidate that fails to parse into the genotype schema after repair → scored as the worst
  candidate seen so far (never dropped from the budget count).
- Any run that touches the evaluation panel during search → discarded and re-run; the
  incident is logged in `STATUS.md`.

## 7. What is not pre-registered

Exploratory: the Twin-2K-500 behavioural block, the API frontier tier, the mutator-LLM
sensitivity study. These are reported as exploratory.

## Amendments

_(none yet)_
