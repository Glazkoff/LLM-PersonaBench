You are a senior ML/ACL researcher brainstorming ideas.

# Task
Generate ideas that make ONE specific submission stronger before the ACL Rolling
Review deadline of 12 October 2026. Roughly 5 weeks. Ideas must be executable in
that window on the compute listed below. This is NOT open-ended blue-sky work.

# The submission
An audit paper on cluster-level LLM personality simulation, auditing a withdrawn
NeurIPS 2026 submission. Established, measured, in the paper already:

A. PROPOSITION. The field's answer-similarity metric S(x,y)=mean(1-|x-y|/4) is
   CRPS with its dispersion term deleted. Define S_lambda = 1 - [E|X-y| -
   lambda*E|X-X'|]/4; lambda=0 is the audited metric, lambda=1/2 is CRPS and is
   the unique proper choice. A faithful sampler is dominated by a constant.
   Measured gap 7.75pp; holds 120/120 items x 4 clusters.
B. AUDIT. On the audited paper's own runs, a constant per-item cluster mean
   scores 0.796 vs the best evolved LLM 0.660 (0.739 after an orientation fix),
   losing 20/20 cells. A real same-cluster human scores 0.721 and ALSO loses.
   At lambda=1/2 the ordering repairs: human 0.859 > constant 0.796.
C. ORIENTATION DEFECT. The released corpus stores human answers reverse-recoded
   on 55/120 items while the simulator answers raw. Correcting raises model-human
   item-mean correlation +0.09 -> +0.63. VR is exactly invariant.
D. REPLACEMENT SUITE. C2ST-AUC, distributional fidelity (W1), variance ratio,
   each against a human ceiling and constant floor. Caveat we disclose: 5
   content-free response-style scalars recover 93% of C2ST separability.
E. RESPONDENT-EQUIVALENTS. A persona is worth ~1 real respondent; the constant
   ~2; the human ceiling ~34.
F. WRONG-PERSONA NULL (80 cells). Conditioning on the right cluster helps
   slightly on location metrics (S +0.021 p=.032, DF +0.026 p=.026) and NOT AT
   ALL on C2ST (p=.51) or VR (p=.55).
G. BELIEF-LEVEL MECHANISM, 6 models, all mass_on_scale=1.000. We read the model's
   own next-token distribution over {1..5} and decompose via the law of total
   variance, VR_total^2 = within + between, each divided by a human
   ACROSS-respondent SD:
      Qwen2.5-7B    7B   within 0.367  between 0.193  total 0.430
      Qwen2.5-32B  32B   within 0.242  between 0.220  total 0.343
      Qwen2.5-72B  72B   within 0.367  between 0.295  total 0.488
      Mistral-24B  24B   within 0.538  between 0.177  total 0.579
      QwQ-32B      32B   within 0.729  between 0.154  total 0.753
      Qwen3-235B  235B   within 0.695  between 0.258  total 0.758
   Rank correlation rho(VR_total, VR_between) = -0.14. QwQ has near-top total
   dispersion and the LOWEST individuation. Within the Qwen2.5 family
   individuation rises monotonically with scale (0.193->0.220->0.295) while
   total does not. No model exceeds VR_between ~0.30.
H. k=4 has silhouette 0.082 (best is k=2) but bootstrap ARI 0.932, collapsing
   after k=5-6. Clusters are not demographic artefacts (Cramer's V 0.055).

# Known open gaps
G1. Three 2026 reasoning-first models (QwQ-family, Qwen3.5-9B, Qwen3.6-35B,
    GLM-4.7) could not be measured under a chat-prefill readout: they emit a
    reasoning preamble, ignore enable_thinking=False, and put ~0.08% of mass on
    the answer tokens. Our diagnostics show the RAW COMPLETIONS endpoint returns
    clean digit distributions for all three. Switching elicitation for everyone
    would change conditioning and require re-running the 6 valid models.
G2. The individuation/uncertainty decoupling is a measurement; nothing optimises
    for it.
G3. The replacement suite is proposed but never used to SELECT a simulator.

# Compute (5 weeks, shared clusters)
- HSE cHARISMa: V100 4x32GB nodes, A100 8x, H100 2x, H200 4-8x. Slurm, busy.
- Euler: 8x H200 143GB, shared with other projects.
- Models cached: Qwen2.5-7B/32B/72B, Mistral-Small-24B, QwQ-32B, Qwen3-235B-FP8,
  and downloaded but unmeasured: Qwen3.5-9B, Qwen3.6-35B-A3B-FP8, GLM-4.7-Flash,
  DeepSeek-V4-Flash-0731 (Jul 2026).
- Open-weight only; logprob access required for the belief readout.

# Closest prior work to beat or cite
- Quantifying the Persona Effect in LLM Simulations (ACL 2024, 2402.10811)
- Stable Personas / temporal stability (2601.22812) -- variance decomposition but
  within model outputs, no human anchor
- Survey Response Generation (2510.11586) -- names token-probability elicitation
- Semantic Similarity Rating (2510.08338, 2607.28550) -- the rival elicitation
- Beyond the Mean: Three-Axis Fidelity (2606.28963) -- no floors/ceilings

# Your task
Generate 8-12 concrete ideas that would make THIS paper materially stronger by
12 Oct. For each:
1. One-sentence summary
2. What it adds to the paper specifically -- which section, which reviewer
   objection it pre-empts
3. Minimum viable experiment on the compute above, with a rough GPU-hour estimate
4. Contribution type: empirical / method / theory / diagnostic
5. Risk LOW/MED/HIGH and effort hours/days/weeks
6. What a NULL result would teach (must be non-empty)

Prioritise ideas that (a) close G1/G2/G3, (b) are cheap relative to their effect
on a reviewer's score, and (c) do not require re-running everything. Be concrete
about what breaks if the idea fails. Do not propose scope that cannot finish in
5 weeks. Say plainly if an idea is not worth doing.
