# Research Idea Report — LLM personality simulation, ARR October

**Direction**: why cluster-level LLM personas are under-dispersed, whether it is
fixable, and what a metric-optimisable simulator looks like.
**Generated**: 2026-09-04. **Fork**: `github.com/Glazkoff/LLM-PersonaBench`, branch `arr-october`.
**Method**: ARIS `/idea-creator`. Codex backend was unavailable (revoked token), so
Phase 1.5 ran on the Tier-2 path — five parallel Claude lens shards
(untested-assumption, method-transfer, diagnostic, contradiction, scaling-regime),
merged and deduped. **41 candidates → 16 distinct after dedup → 4 already executed.**

---

## Executed already (results in hand)

### H3 — Orientation defect ✅ **CONFIRMED, major**
`arr2026/scripts/hyp/h3_orientation_audit.py` · `arr2026/results/h3/`

The corpus stores human answers **reverse-recoded**; the simulator answers the
literal item text. On the **55 negatively-keyed items** the scales run opposite,
and every published model-vs-human number is computed across that flip.

- model↔human per-item mean correlation **+0.092 → +0.633**
- model score **0.6604 → 0.7386** (the simulator was being penalised)
- **VR exactly invariant** (`sd(6-x)=sd(x)`) — under-dispersion untouched
- **constant still wins 20/20**; gap 13.6 → 5.8 pp

Sharpens the paper: corrected, the model (0.739) outscores a *real human* (0.721)
while a classifier separates it at AUC 0.983.

### H1 — The metric is CRPS minus its dispersion term ✅ **CONFIRMED, centrepiece**
`arr2026/scripts/hyp/h1_proper_scoring.py` · `arr2026/results/h1/`

`CRPS(F,y) = E|X−y| − ½E|X−X'|`. The audited metric is the first term alone.
Define `S_λ = 1 − [E|X−y| − λE|X−X'|]/4`; λ=0 is the audited metric, λ=½ is CRPS.
**λ=½ is not tunable — it is the unique proper choice.**

| λ | Model | Constant | Human |
|---|---|---|---|
| 0 (audited) | 0.660 | **0.796** | 0.721 |
| **0.5 (proper)** | 0.697 | 0.796 | **0.859** |
| 1.0 | 0.744 | 0.796 | **0.998** |

λ*(human)=0.30, λ*(model)=0.75; model never overtakes in **17/20** cells.
Gives the paper a principled *fix*, not just a critique.

### H2 — Is C2ST reading traits or response style? 🔄 running (job 99_2)
`arr2026/scripts/hyp/h2_c2st_ablation.py`. Self-critical audit of our own metric:
feature ladder A full items / B 5 style scalars / C ipsatised / D facets / E
forward-vs-reverse. Early cells show style-only AUC already ~0.99 — if that holds,
we must report C2ST as partly style-driven.

### H4 — Variance-loss ladder: belief → decode → parse 🔄 queued (job 102, GPU)
`arr2026/scripts/hyp/h4_variance_ladder.py` + `src/models/providers/local_vllm.py`.
Reads the model's own token distribution over {1..5} via vLLM logprobs.
**Mutually exclusive outcomes, both decisive**: belief SD ≈ human ⇒ loss is at
decoding and a free belief-sampling readout fixes it (and arXiv 2607.28550's
numeric-emission account is wrong on mechanism); belief SD ≪ human ⇒ loss is
representational and no sampling fix can work.

---

## Ranked queue (not yet run)

**Tier 1 — CPU, free, high value**

| # | Hypothesis | Why it matters |
|---|---|---|
| H5 | **Person×item interaction collapse.** Decompose `Y=μ+a_p+b_j+(ab)_pj+e`. Shard measured model item-main-effect 78–95% vs human 26–42%, interaction 4–21% vs 56–77%. | The model doesn't emit "43% of human variance" — it emits ~120% in the wrong component and ~15% in the right one. Reframes the headline. |
| H6 | **Prompt-encoding information ceiling.** The individual is compressed to a **35-symbol code over a 5-letter alphabet** (`get_modifier_by_match`). Compute the oracle ceiling for *any* perfect reader, plus prompt-collision rate. | Bounds what conditioning could ever achieve. May show the ceiling is the binding constraint, not the model. |
| H7 | **Permutation null + placebo persona.** 4×4 cross-evaluation of every persona cluster against every human cluster, plus shuffled-facet placebo and no-persona arms. | No persona paper reports a wrong-persona control, so no persona effect is currently attributable to the persona. |
| H8 | **Respondent-equivalents.** How many real respondents does a persona equal? Predicted single digits. | Converts "loses to a constant" into "loses to five people" — unanswerable by reviewers. |
| H9 | **Metric sample-size bias.** n-sweep 10→1280 for ceiling/floor. C2ST at 40v40 on 120 features is in the overfitting regime; the 0.564 human ceiling may be pure finite-sample bias. | Pre-empts the likeliest reviewer attack and sizes every other sweep. Should run before the GPU sweeps. |
| H10 | **Stability-vs-separation null calibration.** Run E1 on covariance-matched structureless surrogates. | Decides whether bootstrap ARI 0.941 at silhouette 0.082 is evidence of structure at all. |

**Tier 2 — GPU, needs vLLM**

| # | Hypothesis | Why it matters |
|---|---|---|
| H11 | **Temperature frontier + serving determinism.** All seven flagship configs ran at **temperature 0**, so published VR is 100% between-persona. | If one sampling parameter fixes dispersion, a whole literature of embedding machinery is unnecessary. |
| H12 | **Joint-context vs independent administration.** 120-in-one-call vs 12×10 vs 120 independent. | Tests whether self-consistency pressure erases the interaction component. |
| H13 | **k-granularity sweep**, k ∈ {1,2,4,…,512,N}. | Given F5 (stable but not separated), asks what granularity maximises fidelity per unit cost. k=1 and k=N are the anchors. |
| H14 | **Base vs instruct** at matched size (Qwen3 ladder). | The RLHF-collapse story is universally asserted and never tested on this task. |
| H15 | **Latent-draw externalisation.** Make the model emit its own individual deviation before answering. | Turns a flat persona→answers map into the hierarchical draw the task actually requires. |
| H16 | **Distribution-aware prompt evolution.** Swap the GA fitness (`my_evaluator.py:81`) for `S_{1/2}` and for a C2ST-derived objective. | The direct answer to "what does a metric-optimisable simulator look like", and tests whether collapse is **iatrogenic** — caused by optimising a constant-optimal objective. |

---

## Eliminated / folded

| Idea | Reason |
|---|---|
| Free-text + embedding (SSR replication) | Subsumed by H4, which adjudicates the same claim at the logit level for far less compute. |
| CFG persona guidance | Kept but deprioritised — depends on H4's verdict; pointless if the belief is already collapsed. |
| Activation probing / steering | Weeks of effort, needs HF-transformers not vLLM. Out of scope before Oct 12. |
| Conformal coverage suite | Elegant but a second metric contribution; would dilute the paper's single claim. |

## Execution order

1. **H9** first (sizes everything), then H5–H8, H10 — all CPU, all free.
2. **H4** (running) → its verdict selects between H11/H15 (decoding-side) and H14/H16 (training/objective-side).
3. **H16** is the strongest candidate for a second paper, or a §7 of this one.
