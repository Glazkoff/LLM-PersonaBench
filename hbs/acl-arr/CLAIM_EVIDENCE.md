# HBS/A16 — claim → evidence (12 November 2026)

Base manuscript: `Glazkoff/LLM-PersonaBench`, branch `codex/hbs-acl-arr-20261111`, commit `3933f6173dd537ec867d7377b594b6b6fb02e84c`. One central claim: separating person matching, instruction changes and predictor mixing changes what the observed gains establish.

| Claim / estimand | Evidence and locator | Bound on interpretation |
|---|---|---|
| **Matching**: own vs deranged history | Historical primary, main Results 4.1, S2/S4; Qwen/Granite differences 0.04278/0.03225 RPS | Specific historical profiles/readouts; not optimizer superiority or external behavioral validity |
| **Instruction**: T0 vs neutral | R2 direct Granite CAL mixture +0.000197, CI [−0.000541,0.000957]; fixed-half −0.014990. S1 and main external results | Same CAL rule, separately chosen weights; T0 unadapted and mismatched to R2 feature representation. No general adapted-persona conclusion; no equivalence |
| **Mixture/component**: Granite mix vs own decoder | Neutral −0.002886; T0 −0.002689, fixed-pipeline CIs negative. Original Full1400 comparisons −0.00404/−0.00384, original Holm retained | Conditional component gain; CAL adds labels; Full1400 repeats selection and fitting, not a simple refit |
| **LLM necessity**: mix vs strong statistical pooling | Job 27454; Table 1, Fig. 2, S1; neutral/T0 minus stimulus-prior +0.000373/+0.000570, both CIs cross zero | Neither LLM superiority nor equivalence established. Post-exposure controls; library has 33 CAL candidates vs 11 |
| **Calibration uncertainty** | Job 27454, 2000 independent CAL/TEST draws; both Granite vs own decoder intervals cross zero | Reselects mixture weights only; model fits, instructions, denominators fixed. Full training/search/procedure uncertainty NOT estimated |
| **Answer budget** | E3/B1; fixed decoder curve crosses 60-answer hybrid at 11.44; audit 27931 at equal 60 answers: +0.021679 [0.020535,0.022840] | Exposed E3, paired participants; not a lower-budget LLM curve, questionnaire savings or equal total supervision. m<60 LLM contrasts unestimated |
| **Stop/negative evidence** | A18 0/3 continuation gates; S5 retains 18 + 27 exploratory cells | Closed current search line; not universal rejection of evolutionary methods |

**Chronology:** historical frozen comparisons → exposed E3/B1 reanalysis → corrected pre-evaluation R2 freeze → post-R2 direct/component/control analyses → synthetic prospective diagnostics. Internal hash freeze ≠ public preregistration. Replaying the same predictions ≠ independent validation.

**Uncertainty ladder:** fixed fit + participant sampling (central intervals); optional three observed search seeds/orders (limited sensitivity); CAL reselection + TEST sampling (fixed fits); full fitting/search/model-generation variation (unestimated). Do not label the third level “full uncertainty.”

**Evidence path:** local `research-plan-20261102/receipts/20261110/r2-test/AUDIT.json` (27183:14 contrasts/686 endpoints and intervals); controls `A16-ARR-claude-review-20261110/experiment/RESULT.json` (27454); `ops-20261112/paired-answer-audit/RESULT.json` (27931). Public aggregate checker verifies identities only; see `REPRODUCIBILITY.md` for restricted replay boundaries.
