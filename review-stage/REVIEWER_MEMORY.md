# Reviewer Memory

## Round 1 — Score: 4/10 — Verdict: not ready
Backend: codex · reviewer model: gpt-5.6-sol · executor: claude-opus-5 · effort xhigh · difficulty hard
Thread: 01a0811c-ffe9-7ae3-a77c-c874184522b4

### Memory Update (reviewer's own words, verbatim)
- Verify that orientation correction is applied inside H13/H13b and every model-human metric, not merely mentioned in the paper.
- Recheck corrected r, itemwise aligned amplitude, actual-response correlations, and all raw-number interventions.
- Require a formal definition and joint CI for "aligned individuation"; reject product-of-means.
- Track whether "under five per cent" is removed or replaced with corrected values.
- Track whether .482 is renamed/reframed as a supervised conditional-mean reference.
- Require leave-one-item-out or independent-score analysis before accepting H12 efficiency claims.
- Check that corrected DF and respondent-equivalent numbers propagate through E3/H7/H8.
- Require reconstructable per-item artifacts for all 11 models and three interventions.
- Recheck C2ST null calibration, QwQ entropy, the .065 oracle shift, and the page-limit spill.

### Weaknesses raised (ranked, verbatim headings)
1. The H13 alignment analysis repeats the paper's orientation defect  — reviewer's corrected r:
   Qwen2.5-7B .120->.223 | 32B .109->.409 | 72B .157->.523 | Mistral .166->.463 | QwQ .122->.314 | 235B .163->.403
2. r x VR is aggregated incorrectly and is not defined as a human fraction (product of means; proposes
   A = mean_j r_j * sd(M_j)/sd(H_j); gives .043 .050 .077 .056 .030 .078 even uncorrected)
3. The replacement suite also uses mismatched orientations (raw DF .7028 -> corrected .7835; one-human
   plug-in DF .7231, so "one human beats the simulator" is FALSE after correction; C2ST .9828 -> .9943, conclusion survives)
4. The "prompt ceiling" is numerically real but conceptually overframed (.4815 reproduced; but it is a
   supervised conditional-mean decoder, not a bound on arbitrary VR; target item is inside its own facet score)
5. H12's efficiency conclusion does not survive its admitted confound; 11 models are not independent replicates
6. Forty fixed personas, no CIs, and probability tensors committed for only 6 of 11 models
7. C2ST finite-sample story is classifier-specific (max(AUC,1-AUC) biases the null above .5)
8. Smaller defects: "at most .05" is false (Apertus .283->.218 = .065); QwQ entropy stated .63 but committed
   summary says .9682; Conclusion spills onto page 9

### What the reviewer confirmed correct
Proposition + median-set equality; CRPS / lambda=1/2 argument; the .4815 held-out number; between/within/total
variance for committed tensors; VR invariance to reverse scoring; C2ST conclusion survives correction;
the raw intervention's VR directions match its summaries (only the alignment reading is invalid).

---

## Round 2 — Score: 5/10 — Verdict: not ready
Backend: codex · reviewer model: gpt-6-astra · executor: claude-opus-5 · effort xhigh · difficulty hard
Thread: 01a08143-c37e-74f2-bf77-a983b9ae5394 (fresh thread; model changed at user request, memory carried by this file)

### Independently reproduced by the reviewer (confirms round 1's fixes were genuine)
corrected r/A for all six reconstructable models; oracle SD reference .481538; oracle-actual r .490891;
QwQ entropy .968245; corrected E3 DF .783479 and C2ST .994288; paired S .738618 vs constant .796107, 20/20.

### Memory Update (reviewer's own words, verbatim)
- Round 2: 5/10; not ready. Local Codex review; same-family; acceptance status provisional.
- New central issue: H1/H2/H5/H7 retain orientation defects. Corrected H1 CRPS .772218; models beat constant
  in 2/20 cells at lambda=.5 and 17/20 at lambda=1. Corrected H7 gaps: S .064752, DF .076701.
  Corrected H5 total variance ratio .942187.
- A is human-SD-normalized amplitude, not human-baseline-normalized fidelity: actual-human A=.497805.
  Qwen72 A=.183071, conditional persona-bootstrap CI [.1677,.1927], B=2000, seed=260908.
- Oracle "ceiling", 35-versus-10 channel mismatch, target-containing H12 denominators, and causal efficiency
  claims remain unresolved.
- Five model tensor sets plus three intervention tensor sets remain absent.
- C2ST folding biases the null; n=20 failure disappears when minimum leaf size changes from 20 to 5.
  H7 permutations ignore shared structure; H8 calibration uses mismatched sample sizes.
- Temperature can restore marginal spread; do not infer that it restores fidelity. Correct SD/variance terminology.
- Conclusion still spills onto page 9. Preserve the valid proposition and surviving empirical audit; require
  complete numerical propagation before another readiness verdict.

### Fixed in response to round 2 (this round)
- table_metricsuite.tex was stale (DF .703 / C2ST .983) after the prose was corrected -> now .783 / .994;
  abstract C2ST updated to .994.
- A normalization: verified A_human = .4978 and A_oracle = .4815 independently; body and conclusion now
  report A against those references (best model 37% of the human reference, weakest 7%).
- Temperature impossibility claim was FALSE; verified p^(1/T) raises Qwen2.5-7B from .43 to 1.02 at T=5.
  Restated as adding dispersion without adding alignment.
- m=2 plug-in (.802) had been conflated with the constant (.796) in my own round-1 correction. Separated.
- Reproducibility moved to the appendix; content now ends p8.

### Still outstanding (NOT fixed)
H1/H2/H5/H7 orientation propagation and all dependent tables; joint persona bootstrap CIs on A;
renaming the .482 reference and the 35-vs-10 channel mismatch in the main text; H12 causal reading;
missing probability tensors for 5 models and 3 interventions; C2ST folding/leaf-size calibration;
H7 permutation structure; H8 sample-size mismatch; SD-vs-variance terminology.

---
