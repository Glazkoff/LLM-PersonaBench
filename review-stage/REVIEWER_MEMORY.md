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

## Round 4 — Score: 6/10 — Verdict: Almost (would not submit current PDF)
Backend: codex · gpt-6-astra · thread 01a08589-e5f8-78f3-923f-7d9f7c81ba53

Round 4 confirmed all four prior blockers cleared or partially cleared, and independently
reproduced all 14 bootstrap intervals, the 56 valid tensors, and the H2 ladder aggregates.

### Fixed since round 4 (verify)
1. FIGURE 1 — was still .654/.983 in the built PDF. Root cause: make_paper_assets.py wrote to
   ROOT.parent/"paper", a sibling directory the submission never reads, so every prior
   regeneration was a no-op. Path fixed; assets rebuilt on the cluster from corrected results;
   figure now carries .739 and .994 (verified by extracting text from figures/inversion.pdf).
2. NOISE FLOOR — inference deleted. Correctly stratified: .54-.77 within-cluster, .23 on the
   four-cluster average, 1.33 only when clusters are pooled. Kept as descriptive context with
   an explicit note that baseline LEVEL variation does not bound a PAIRED gain.
3. H8 INTERVALS — equivalence argument withdrawn; m=2 retained as a mean-curve threshold;
   human ceiling now "not crossed on the evaluated grid up to m=500" rather than off-curve.
4. H6 — "a perfect reader gains under a point" deleted; reported as one fitted decoder's score.
5. GLOBAL MEAN — intro corrected to 19/20 (GPT-4.1-nano c1 = .8066 > .7961).
6. HEAD-TO-HEAD CAPTION — "changes no ordering" was false; now "preserves all 20 matched
   constant wins, though it does reorder the models".
7. FP8 CONTRADICTION — Limitations claimed every reported model is unquantised at inference,
   but Qwen3-235B's checkpoint is ...-2507-FP8 served at FP8. Precision now stated per model.
8. H12 — "every model" -> 10/11, Qwen3-235B the exception (.224 named vs .264 elsewhere).
9. UNDERCLAIM CORRECTED — paired differences on the same joint draws:
     Qwen3-235B - GLM-4.7      +0.0172 [+0.0095,+0.0249]  resolved
     Qwen2.5-32B - Granite-4.1 +0.0340 [+0.0278,+0.0400]  resolved
     Granite raw - orig        +0.0516 [+0.0446,+0.0589]  resolved
     Apertus raw - orig        -0.0124 [-0.0158,-0.0089]  resolved
     gpt-oss raw - orig        +0.0083 [+0.0048,+0.0119]  resolved
   All match round 4's independent computation. My withdrawal of the GLM/235B claim on
   marginal overlap was a statistical error; it is restored on the paired interval, and the
   intervention is now "every model moves resolvably, in model-dependent direction".
10. h13c reports the ORIGINAL-SAMPLE estimate with the percentile interval, not the bootstrap
    mean (which is what produced .183 next to .180).
11. Conclusion reports 37% [34,40] rather than "at best a third" (underclaim).
12. n-floor qualified by classifier configuration in main text as well as appendix.
13. H2 80% restated as attainment by style features, not an attribution decomposition.
14. Split-sensitivity moved to an appendix now that only descriptive content remains.

Content ends p8 under the boundary check; PDF is 12 pages total.

---

## Round 5 — Score: 7/10 — Verdict: Almost ("only item 1 blocks uploading")
Backend: codex · gpt-6-astra · thread 01a085a0-762b-7713-893f-68da73f2908a

Round 5 confirmed in the BUILT PDF: Figure 1 shows .739/.994; noise-floor inference gone with
correct stratification (.541/.574/.743/.770 within, .231 average, 1.329 pooled); H8 m=2 is a
mean-curve threshold; H6 bound removed; Conclusion genuinely ends p8. Independently reproduced
20/20 constant wins, 19/20 global-mean wins, H1 2/20 at CRPS, H5 .9422, all 56 tensors, all 14
marginal and all 5 paired intervals. Confirmed 37% [34,40] is supported (36.8% [34.3,39.6]).

### Fixed since round 5 (verify)
BLOCKER (item 1) — both parts:
  - decoder table overflowed 22.52pt on p11; retyped with p{0.72\columnwidth}r; zero large
    overfull boxes remain.
  - \usepackage[]{acl} -> \usepackage[review]{acl}; verified in the built PDF (lineno loaded,
    consecutive line numbers on p1, "Anonymous ACL submission" header).
NON-BLOCKING:
  - original-sample estimates used consistently (human .498, Qwen2.5 .067/.116/.183,
    Apertus .033); paired differences quoted at original-sample values.
  - the asset generator's INPUT csv was still the stale 18-cell e3 file, so regenerating from
    the repo would have restored .654/.983. Corrected 20-cell CSV committed at that path;
    generator now reproduces .739/.994 from the repository alone (verified locally).
  - H10 outputs supplied.
  - reproducibility claim narrowed: k=4 ARI differs in the 4th decimal (.9463 vs .9509).
  - H6 no longer "faithful"/"reproduces the conditional distribution"/"strictly closer to the
    truth"; "no model supplies individuation" -> "what stays limited is individuation".
  - silhouette not "monotonic"; Fig 2 caption says SD ratio not variance; "also also" fixed;
    abstract says evolved personas average .739.

---

## Round 6 — Score: 7/10 — Verdict: READY FOR SUBMISSION: Yes
Backend: codex · gpt-6-astra · thread 01a085b0-dfc0-7223-b52b-28c66cfdc7f1

"The round-5 upload blocker genuinely clears. I found no remaining BLOCKING issue."

Verified in the built PDF: review mode active (anonymous heading + line numbers), decoder table
fits on p11 without collision, Conclusion finishes on p8 before Limitations, zero overfull-box
warnings. Independently reconstructed: corrected paired S .7386215; 20/20 constant and 19/20
global-mean wins; E3 20 finite cells with C2ST .9942875 and DF .7834792; H1 2/20 wins at
lambda=.5; H5 .9421871; human aligned-amplitude reference .4978048; Qwen2.5 .0671/.1155/.1831;
best model 36.78% [34.33, 39.58] of the human reference. All 56 valid tensors validated; all 14
marginal and 5 paired intervals reproduced; asset regeneration reproduces all three tables and
both embedded figures from the repository alone.

Four NON-BLOCKING items, all since fixed:
  - categorical wording ("predicting the distribution correctly is penalised"; "not who endorses
    them") -> restoring marginal spread / respondent-specific alignment remains limited
  - h13c printed bootstrap means for the human reference and paired differences -> original-sample
    estimates with bootstrap intervals throughout
  - three double-rounding errors (lambda=.75 model .795 not .796; appendix constant DF .795 not
    .796; Qwen3-235B/GLM paired lower endpoint +.009 not +.010) and my incorrect "fourth decimal"
    description of the two k=4 ARIs (.946 vs .951 differ at the third)
  - Figure 2 axis said "variance ratio" -> SD ratio; Dong et al. entry marked [VERIFY] rather than
    given an invented arXiv id (DBLP/CrossRef unreachable from this environment)

Reviewer's standing caveats: review independence is same-family; acceptance status provisional;
external bibliography not independently verified.

---

## Between rounds 6 and 7 — acting on the "what would make this an 8" guidance

Round 6's follow-up ranked five routes. I executed rank 1 (CPU, existing tensors) in full.

H14 (`arr2026/scripts/hyp/h14_conditional_crps.py`, results in `arr2026/results/h14/`):
scores four predictors on the same respondents under normalised CRPS
L(Q,y) = (1/4) sum_t (q_t - 1{y<=t})^2 --- correct assignment, rows permuted within
cluster (identical mixture, destroyed identity), the cluster empirical prior, and a
supervised conditional CDF fitted on held-out respondents.

Across all 11 models:
  identity gain (permuted - correct)  +0.0035 to +0.0173, EVERY paired interval excludes 0
  versus the cluster prior            -0.0360 to -0.0952, every interval excludes 0
  decomposition                       V = 0.0227 (conditioning value, constant)
                                      E = 0.0584-0.1199 (model prediction error)
                                      E/V = 2.6x to 5.3x
  identity check                      prior - correct = -0.0827 vs V - E = -0.0840 (Qwen72)

Both ranges independently reproduce round 6's own exploratory numbers (.0035-.0173 and
.036-.095), which it computed before I ran this.

New central claim, now in abstract, body and conclusion: every model reads the persona in the
right direction AND still loses to ignoring it, because prediction error exceeds the value
conditioning carries. This replaces "models do not individuate", explains the earlier VR and
alignment results instead of sitting beside them, and rests on a control (row permutation)
that no population-level metric can see through.

NOT yet done from the follow-up: rank 2 (SD3 behavioural replication with disjoint
conditioning/target items), rank 3 (fixed-candidate selection experiment), rank 4 (elicitation
format), rank 5 (independent pipeline). Also not done: the conditional-median characterisation
of constant-optimality.

---

## Round 7 — Score: 7.5/10 — Verdict: Almost ("with these corrections, 8/10 and Yes")
Backend: codex · gpt-6-astra · thread 01a085de-ca28-7c13-99cc-fb7cf7fcb900

Verified independently: CRPS normalisation correct (agrees with the absolute-distance form to
<5e-9); orientation correct; all 44 evaluated tensors finite simplexes; decoder trained on
respondents disjoint from the 40 evaluated; permutation control valid. With 10,000 paired
cluster-stratified bootstrap draws recomputing the control, all 11 identity intervals stay
positive and all 11 prior comparisons stay negative.

### Fixed since round 7 (verify)
BLOCKER 1 — fitted reference presented as exact. Replaced with the identity round 7 supplied
and verified, in measured quantities only:
    R(prior) - R(correct) = I - D_Q - [R(qbar) - R(prior)]
Now exact to machine precision: residuals |.|<6e-17 for all 11 models (was 1.4e-3 for Qwen72).
The 2.6x-5.3x ratio is GONE from the paper. The exact decomposition also changes the reading:
mixture excess 0.038-0.090 dominates, I is 0.004-0.017 and D_Q 0.002-0.021, so the loss sits in
the population term and identity effects are an order of magnitude smaller. My earlier framing
overweighted identity because the fitted reference concealed where the loss was.
BLOCKER 2 — decoder was not a CDF (449/19200 non-monotone). Now projected onto the monotone
cone by pool-adjacent-violators; loss effect negligible as round 7 predicted.
NON-BLOCKING — permutation control computed as an exact expectation over all donor rows, not
Monte Carlo; central claim scoped to "a cluster empirical prior fitted on other respondents";
Remark 1 no longer claims the optimum is a point rather than a distribution, since randomisation
on the conditional median set ties.

Still not done: SD3 behavioural replication with disjoint conditioning/target items (round 7's
next substantive priority, explicitly NOT required for 8), ranks 3-5 of the earlier route.

---

## Round 8 — Score: 8/10 — Verdict: Almost (one BLOCKING manuscript-consistency item)
Backend: codex · gpt-6-astra · thread 01a085ea-115a-75e2-8212-6892ae4ca819

Verified independently: the observable identity reproduces to floating point for all eleven
models, and follows algebraically from E_pi R(Q_pi) = R(qbar) + D_Q. Decoder monotonicity
violations 449/19200 -> 0/19200, PAVA checked against enumeration of all contiguous partitions.
10,000 joint cluster-stratified bootstrap draws recomputing the donor control: all identity
intervals positive, all prior comparisons negative. Review formatting, content on p8.

Round 8 caught that THREE corrections I had reported as applied were absent from source and PDF.
It was right: the batch containing them aborted on a failed assertion for a fourth edit, which
correctly prevented a partial write, and I then reported the batch as done without checking.
Now applied and each verified in the BUILT pdf:
  - abstract retired explanation ("prediction error exceeds the value conditioning carries")
    replaced -- it named two quantities the replacement identity does not contain
  - Remark 1's "the winner is still a point, not a distribution" removed; a deterministic
    conditional median is always optimal and randomisation ties when supported on that set, so
    absolute error never UNIQUELY rewards faithful sampling
  - body comparator scoped to "a cluster empirical prior fitted on other respondents", noting it
    uses cluster membership but no persona
Plus round 8's item 3: mixture dominance now stated precisely as |D_Q - I| < 0.005 for every
model and below a tenth of that model's mixture excess (verified, max ratio 0.095), instead of
the vaguer "order of magnitude smaller".
h14 docstring updated: it had still described the discarded fitted-reference decomposition as
exact.

Still open and NON-BLOCKING per round 8: SD3 behavioural replication with disjoint
conditioning/target items; H14 interval conditioning could be described more fully.

---

## Round 9 — Score: 8/10 — Verdict: READY FOR SUBMISSION: Yes
Backend: codex · gpt-6-astra · thread 01a085f2-ef50-7f23-980b-399ce9f9e6f1

"No remaining BLOCKING issue found. I would submit this version."

All five corrections verified in the BUILT PDF this time (abstract p1 lines 039-043; Remark 1
p2 lines 129-145; body comparator p7 lines 539-541; mixture-dominance statement p7 lines
551-556; h14 docstring in the script). Independent reconstruction from all 44 tensors:
  D_Q - I  range  [-0.004471224, +0.004997033]
  max |D_Q - I| / mixture excess = 0.094160955
  max decomposition residual = 1.8e-15
Conclusion ends p8; review anonymity and line numbering active.

Applied after round 9 (its two optional/non-blocking items):
  - "in absolute value" added before "below a tenth", per its clarity note
  - appendix paragraph documenting H14's interval construction: 2000 percentile-bootstrap
    draws over respondents, 40 per cluster with equal cluster weight, permutation control
    computed exactly rather than sampled, prior and decoder fitted once on excluded
    respondents and held fixed, so intervals cover respondent sampling and not fitting

Remaining NON-BLOCKING, carried into the submission as stated limitations:
  - SD3 behavioural replication with disjoint conditioning/target items (next substantive step)
  - the Dong et al. bibliography entry remains [VERIFY]; DBLP/CrossRef unreachable from here

Reviewer's standing caveats: same-family review independence; acceptance assessment provisional;
external bibliography not independently verified.

---
