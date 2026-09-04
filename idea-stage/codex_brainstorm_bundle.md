You are a senior ML researcher brainstorming research ideas.

# Research direction
Cluster-level LLM personality/survey-respondent simulation, and its evaluation.
Target venue: ACL Rolling Review October cycle (deadline 12 Oct 2026) -> NAACL/COLING 2027.

# ESTABLISHED FACTS (our own audit; treat as given, do not re-derive)
Corpus: IPIP-NEO-120, 410,168 respondents, 4 k-means clusters over 30 facet scores.
Systems audited: 5 LLMs (Qwen3-235B-A22B, GigaChat3-10B, GPT-4.1-mini, GPT-4.1-nano,
Grok-4.1-fast) x 4 clusters, evolutionary prompt optimisation of persona prompts.
Metric under audit: S(x,y) = mean_j (1 - |x_j - y_j| / 4) over 120 Likert items.

F1. PROVEN. Under S, a simulator that samples faithfully from the target cluster's
    per-item response distribution is STRICTLY DOMINATED by the best constant predictor.
    Constant scores 1 - MAD/4 (mean abs deviation about the median); faithful sampler
    scores 1 - GMD/4 (Gini mean difference); GMD >= MAD always. Measured gap 7.75pp,
    holds on 120/120 items in all 4 clusters. (The GMD>=MAD inequality itself is
    classical - Yitzhaki & Lambert 2013 - the novelty is that it governs this metric.)
F2. Constant per-item cluster mean scores 0.796; best evolved LLM scores 0.660.
    LLM loses in 20/20 cluster x model cells. Global mean (ignores clusters) 0.720,
    also beats every LLM cell.
F3. A REAL human drawn from the target cluster scores 0.721 and ALSO loses to the
    constant in 20/20 cells. The metric ranks a constant above a genuine group member.
F4. Our replacement suite over the same 20 cells:
      C2ST-AUC (0.5=indistinguishable): model 0.983, human ceiling 0.564, constant 0.926
      Distributional fidelity (W1-based): model 0.703, human 0.939, constant 0.795
      Variance ratio (1.0 ideal):        model 0.431, human 1.018, constant 0.000
    => LLM personas emit ~43% of human response variance. UNDER-DISPERSION is the
    central unexplained phenomenon.
F5. k=4 is unsupported by any internal criterion (silhouette falls monotonically
    0.134@k=2 -> 0.082@k=4 -> 0.053@k=15; Davies-Bouldin minimal at k=2; gap statistic
    selects no interior k) BUT bootstrap ARI at k=4 is 0.941 - the partition is
    STABLE yet NOT SEPARATED. It discretises a continuum reproducibly.

# LANDSCAPE (2026, verified by search)
- "Correcting Mode Collapse in Silicon Sampling with Semantic Similarity Rating"
  (arXiv 2607.28550): claims mode collapse stems from LLMs being bad at emitting
  NUMERIC scores; generating free text and mapping to the scale via embeddings
  improves distributional fidelity. A direct rival explanation for our F4.
- "Beyond the Mean: Three-Axis Fidelity..." (arXiv 2606.28963): proposes structural /
  marginal / individual fidelity axes. Explicitly does NOT report constant-predictor
  baselines or human ceilings - a stated methodological gap.
- "Distribution-First Population Simulation: Collapse, Calibration, and Recall in
  Non-WEIRD LLM Persona Modeling" (arXiv 2607.18310).
- "Distribution Shift Alignment Helps LLMs Simulate Survey Response Distributions"
  (arXiv 2510.21977).
- "Assessing the Reliability of Persona-Conditioned LLMs as Synthetic Survey
  Respondents" (arXiv 2602.18462).
- RLHF/preference collapse is a suspected cause of narrowed output distributions.

# GAPS WE SEE
G1. WHY are personas under-dispersed? Competing causes untested against each other:
    decoding temperature; RLHF/instruct-tuning; numeric-vs-text output format;
    single-prompt-per-cluster (no within-cluster individual variation); the model
    answering as "the average member" because the prompt describes a centroid;
    item-order/context effects; refusal/hedging toward the midpoint.
G2. Is under-dispersion FIXABLE, and at what cost to central tendency? Nobody has
    swept the variance/location trade-off explicitly.
C3. What does a simulator optimised AGAINST a distribution-aware metric actually do?
    All prior optimisation targets pointwise agreement.
G4. Nobody reports a constant floor and human ceiling as mandatory anchors.
G5. Cluster-conditioning may carry almost no signal (F5). Untested: does persona
    conditioning beat NO conditioning under distribution-aware metrics?

# COMPUTE AVAILABLE
Euler cluster: 8x H200 NVL 143GB, Slurm (partitions train/infer), 768 CPU cores,
3TB RAM. OPEN-WEIGHT MODELS ONLY (Qwen3-235B-A22B-Instruct-2507 via vLLM TP=4,
GigaChat3-10B-A1.8B). Proprietary APIs are deliberately out of scope for now.
Existing code: LLM-PersonaBench (OpenAI-compatible provider layer, so a local vLLM
server drops in). ~5 weeks to deadline.

# YOUR TASK
Generate 10-14 concrete, TESTABLE research hypotheses that go BEYOND the audit above.
Prioritise ones that explain G1 (why under-dispersion), test G2 (is it fixable), and
define G3 (what a metric-optimisable simulator looks like).

For each idea give:
1. One-sentence summary
2. Core hypothesis - what you expect and WHY (the mechanism)
3. Minimum viable experiment - cheapest decisive test, on the compute above
4. Contribution type: empirical finding / new method / theoretical result / diagnostic
5. Risk: LOW / MEDIUM / HIGH
6. Effort: hours / days / weeks
7. What a NULL result would teach (this must be non-empty - we want ideas that are
   informative whichever way they land)

Constraints: open-weight models only; must be runnable in <= 2 weeks of wall-clock on
the cluster above; prefer one mechanism with few moving parts. A decisive negative is
as valuable as a positive. Be genuinely creative - inverted assumptions, questions
nobody thought to ask - but every idea must be falsifiable with the resources listed.
