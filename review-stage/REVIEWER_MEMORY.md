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
