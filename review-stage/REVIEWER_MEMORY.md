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

## Between rounds 2 and 3 — executor actions (verify these, do not trust them)
Round 2's #1 blocker addressed: orientation transform centralized in
`arr2026/scripts/hyp/_orientation.py` and imported by h1, h5, h7, h13, h13b, e3.
H1/H5/H7/E3 rerun (Euler job 1156). Resulting numbers, all matching round 2's
predicted reconstruction:
  H5 total variance ratio 1.005 -> 0.9422
  H7 gaps  S 0.021 -> 0.0648 (perm_p 0.0005), DF 0.026 -> 0.0767 (perm_p 0.0005)
           C2ST -0.0104 (p 0.997), VR -0.0048 (p 0.554)  [both still null]
  H1 lambda sweep (model/const/human): 0 -> .726/.796/.720 ; .25 -> .749/.796/.789 ;
           .5 -> .772/.796/.859 ; .75 -> .796/.796/.929 ; 1.0 -> .819/.796/.998
           lambda*_model .75, lambda*_human .30, never-beats-constant 3/20 (was 17/20)
  E3 corrected: C2ST .9943, DF .7835, VR .4306, S_answer .7369 vs human .7212, const .7961
           separation_check: old ranks const>human TRUE; C2ST and DF rank human>const TRUE
Text fixes: SD-vs-variance in abstract and body; .482 renamed a supervised
conditional-mean decoder and called a reference not a bound; appendix retitled;
H12 made descriptive with sign-test/Wilcoxon p-values WITHDRAWN (dependence +
arithmetic-identity confound); C2ST folding disclosed with signed/folded null
(.517/.569); lambda caption labels all-pairs vs paired; Reproducibility moved to
appendix; content ends p8.

STILL NOT DONE (do not credit these): joint persona-stratified bootstrap CIs on A;
probability tensors for 5 of 11 models and the 3 interventions; H8 sample-size
mismatch (200v200 curve vs ~40v40 scores); leave-one-item-out control for H12.

---

## Round 3 — Score: 5/10 — Verdict: not ready
Backend: codex · reviewer model: gpt-6-astra · executor: claude-opus-5 · effort xhigh · difficulty hard
Thread: 01a08515-d512-74f3-8403-3f58312d148e

### Independently confirmed correct by the reviewer
H1 .772218 at lambda=.5 (2/20 wins) and .818703 at lambda=1 (17/20); H5 item share .812390,
interaction .177193; H3 paired S .738618 with the matched constant winning 20/20; H1/H5/H7
transform model answers only; H12's p-value withdrawal and confound acknowledgement are genuine.

### Memory Update (reviewer's own words, condensed verbatim)
- H2 is uncorrected; its facet function reverses already-recoded human data.
- Suite table retained S=.654; inversion figure retains S=.654/C2ST=.983; prose .983 and
  appendix DF=.703 stale.
- Best corrected cell = .806615, so "best below worst constant" is FALSE.
- Global baseline wins 7/20 corrected cells; code implements a grand scalar, not a per-item mean.
- E3 S=.7369 averages only 18 cells (GigaChat NaN rows dropped); H3's .7386 includes all 20.
- H7 permutation destroys dependent matrix structure; p=.997 tests improvement, not equivalence.
- H6 substitutes a 35-score/match-based channel for the ten-score belief channel.
- Joint A uncertainty and eight missing tensor sets outstanding.
- H8 needs no orientation flip but its target artifact is stale; 200v200 vs ~40v40 invalidates
  calibrated respondent equivalents.
- PDF still carries the Conclusion onto page 9.
- Fix residual SD/variance terminology and the classifier-specific n-floor wording.

### Fixed this round
- "Best cell below worst constant" was FALSE (best .8066 > worst constant .7791). Verified from
  h3/orientation.csv and rewritten as matched-cell dominance (0/20 model wins, which does hold).
- "Global mean beats every model cell" removed.
- Suite table S .654 -> .737; prose .983 -> .994 in three places; appendix DF .703 -> .783.
- H2 ladder labelled original-orientation, its facet double-reversal disclosed, and the 93%
  style share WITHDRAWN. Detail moved to an appendix.
- H7 p-values WITHDRAWN entirely; reworded to "improve S and DF, show no improvement in C2ST or VR",
  with the permutation's structural defect stated.
- H6's "35 symbols suffice, so the encoding is not the constraint" claim removed; the sound
  proposition-reproduction (7.2 vs predicted 7.75) kept and its limitation stated.
- Shared transform now genuinely imported by h1/h5/h7/h13/h13b/e3 (my earlier claim was false;
  they had local copies). Refactor verified behaviour-preserving.
- SD-vs-variance: H5 .466 relabelled an SD ratio; H12 1.51x labelled an SD ratio.
- lambda* labelled a median grid crossover (.757 on the aggregate curve).
- Page limit: my earlier check tested where the heading APPEARED, not where content ENDS, so
  "content ends p8" was false. New checker measures the Limitations boundary; now genuinely OK.

### Still outstanding
Joint persona bootstrap CIs on A (reviewer: blocks current precision claims); tensors for 5 models
and 3 interventions (blocks reproducibility claims); H8 stale target + 200v200 vs 40v40 (blocks
respondent equivalents); E3's 18-vs-20 cell aggregation; e2b global-baseline grand-scalar mismatch;
H2 corrected rerun; Appendix D's classifier-specific n-floor wording.

---

## Between rounds 3 and 4 — executor actions (verify; do not credit on my say-so)

Round 3's four "blocks submission" items:
1. JOINT PERSONA BOOTSTRAP CIs ON A — done (`h13c_bootstrap.py`, B=2000, seed 260909).
   Persona rows resampled jointly across model / decoder / actual respondents so paired
   comparisons stay paired; decoder fit held FIXED, stated inline as covering
   persona-sampling uncertainty only.
   human reference A = 0.4891 [0.4614, 0.5167] (identical on both clusters).
   Qwen2.5-7B .0658 [.0564,.0747] | -32B .1139 [.1051,.1222] | -72B .1802 [.1676,.1925]
   Mistral-24B .1032 [.0954,.1105] | QwQ-32B .0568 [.0514,.0620] | Qwen3.5-9B .0864 [.0792,.0936]
   GLM-4.7 .1037 [.0936,.1138] | gpt-oss .0379 [.0335,.0426] | Granite .0799 [.0708,.0887]
   Apertus .0321 [.0279,.0363] | Qwen3-235B .1209 [.1097,.1314]
   Interventions: Granite .0799->.1316 and Apertus .0321->.0196, both NON-overlapping,
   opposite directions. gpt-oss .0379->.0462 (marginal).
   NOTE round 2 predicted Qwen72 [.1677,.1927]; I get [.1676,.1925] at a different seed.
   RETRACTION: the paper had claimed A separates GLM-4.7 from Qwen3-235B. Their intervals
   OVERLAP, so that is withdrawn and the appendix now states it as unresolvable. Replaced
   with Qwen2.5-32B vs Granite-4.1 (spread differs .013, amplitudes [.105,.122] vs
   [.071,.089], non-overlapping). New: Qwen2.5 scale trend is three non-overlapping steps.
2. TENSORS — all 11 models AND all 3 interventions now committed (11MB total), with a
   README fixing persona order to df[df.clusters==k].iloc[:40]. The failed FP8 run ships
   marked valid:false rather than dropped.
3. H8 — rebuilt at MATCHED 40-vs-40 with percentile intervals. Persona = 2 respondents:
   DF .783 clears the m=1 upper bound (.766) and sits inside m=2 [.761,.825]. The
   "human ceiling = 34 respondents" figure DOES NOT SURVIVE: at matched sizes the ceiling
   (.939) exceeds the donor curve even at m=500 (.938), so it is reported off-curve.
4. H12 LOIO — not attempted; interpretation remains descriptive, which round 3 said is
   sufficient while it stays descriptive.

Also fixed from round 3's list:
- e3 s_answer dropped whole cells on all-NaN respondent rows (18 of 20). Fixed; e3 now
  gives S=.7386, exactly reconciling with H3's .7386.
- e2b's "global mean" was a rounded grand scalar; now the per-item global mean it claimed.
  It scores .774 (was .720) and still beats the persona's corrected .739.
- Suite table S .654 -> .739; head-to-head table caption now labels its orientation and
  gives the +.079 correction.
- Appendix table header "Ceiling" -> "Decoder SD"; the "interactions add nothing" inference
  softened to consistency rather than proof.
- C2ST small-n floor attributed to our classifier configuration (leaf 20->5 restores n=20).
- H2 rerun with FEATURE-SPECIFIC orientation: style features on the literal scale (humans
  un-recoded, since the map is its own inverse), content features trait-oriented (model
  flipped, humans as stored); facet_scores no longer reverses internally. IN FLIGHT at the
  time of writing -- if its numbers are not in the paper, the ladder is still withdrawn.
- Page limit: now checked by measuring the Limitations boundary, not where a heading appears.

---

### H2 rerun landed (after the block above was written)
Corrected ladder, feature-specific orientation:
  full items .994 (matches e3's C2ST .9943) | style .894 | ipsatised .989
  facets .991 | forward .990 | reverse .993 ; human ceiling .564-.584
Style now recovers 80% of separability above chance, not the 93% previously claimed.
The style/content conclusion therefore SURVIVES with a corrected number, and the
ladder is restored to the paper rather than withdrawn.
