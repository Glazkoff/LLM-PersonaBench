# Paper Improvement Log

## Score Progression

| Round | Score | Verdict | Key changes |
|-------|-------|---------|-------------|
| Round 0 (original) | 4/10 | No | Baseline |
| Round 1 | pending | — | Three verified factual errors fixed; see below |

## Round 1 — reviewer: gpt-6-astra (xhigh), fresh thread `01a09f52-7684-70f1-bed7-a78b3ae49e50`

Score 4/10, verdict **No**. The reviewer read the source, the rendered PDF and
the saved result files, and found errors that were checkable against code. Three
were confirmed against the artifacts and fixed first, because they are integrity
defects rather than matters of framing.

### CONFIRMED ERROR 1 — a table labelled "Bonferroni CI" carried ordinary 95% bounds

`h17_selection.py` computed the simultaneous bounds (`blo, bhi`), used them
correctly in the resolved/unresolved verdict, and then printed `lo, hi`. Those
printed numbers were transcribed into App. D under a Bonferroni heading.

Fixed by printing both intervals plus the original-sample point estimate, and
replacing the table. The conclusions are unchanged: all five resolved
comparisons remain resolved under the true simultaneous bounds.

  s0          bootstrap +0.294   95% [+0.250,+0.362]   Bonferroni [+0.247,+0.365]   point +0.256
  mode_acc    bootstrap +0.267   95% [+0.223,+0.282]   Bonferroni [+0.098,+0.287]   point +0.271
  ev_mae      bootstrap +0.032   95% [+0.016,+0.052]   Bonferroni [+0.008,+0.059]   point +0.034

### CONFIRMED ERROR 2 — a mean reported as a maximum

App. F claimed every aggregate moves "by at most 5e-3" across GPU
architectures. 4.7e-3 is the MEAN over four cells; the largest single-cell
change is 1.62e-2 (S_1/2, gemma-4-E2B/SD3). Corrected in both the appendix and
Limitations.

### CONFIRMED ERROR 3 — a ratio across incompatible units

The same paragraph divided raw score changes by candidate-range-normalised
regrets to claim a "six to sixty times" robustness margin. Those are different
units. The ratio is removed and replaced with the direct statement: a change of
architecture moves a model's score by at most 0.016 while moving 7.7% of its
individual answers.

### Outstanding from Round 1 (not yet applied)

- MAJOR: selectors are graded on different evaluator sets, so the same selected
  model can receive different regrets. Needs a common-evaluator comparison.
- MAJOR: conditioning information (cluster centroid vs respondent-specific
  modifiers) is not stated consistently; Prop. 1 applies to the unconditional case.
- MAJOR: S_{1/2} is defined as a reward in Eq. 2 but used as a loss in tables.
- MAJOR: mechanism claims in the intro exceed what App. E supports.
- MAJOR: audited system is not cited; no artifact link.
- MINOR: abstract/intro open on background; progress-report narration; defensive
  asides; Figure 1 legibility.

## Round 2 — reviewer: gpt-6-astra (xhigh), fresh thread `01a09f66-2e8e-7a42-b476-2d39a62a5efb`

Score **5/10** (from 4/10), verdict **No**. Again the reviewer recomputed from
the artifacts; the central proposition and headline audit reproduced exactly
(constant 0.799558 vs faithful 0.722081, gap 7.747759, strict on all 480
cluster-item combinations; corrected model 0.738618 vs constant 0.796107,
20/20 and 19/20).

### CONFIRMED ERROR 4 — the released selection record did not match the paper

I refreshed the paper from the common-evaluator rerun but left
`selection_regret_n512_12cand.txt` holding the previous run, so the artifact
and the table disagreed on every row. Line 964 also still quoted the old mode
accuracy (+0.267) against its own table (+0.238). Both fixed; the file now
comes from the same job as the table.

### CONFIRMED ERROR 5 — h19_dispersion.py had no orientation correction

`h17_selection.py` flips the model's simplex on reverse-keyed items when the
corpus stores recoded answers; `h19_dispersion.py` did not, so every score it
computed on IPIP used mis-oriented targets. This is the same defect that
invalidated an earlier round of this project, reintroduced in a new script.

Corrected, the IPIP result changes materially: the S_0/dispersion correlation
goes from -0.02 to +0.03, and S_0's pick changes from gpt-oss-20b to
Qwen3.6-35B-A3B -- the same model the proper rule picks.

The paper had claimed the correlation was "negative on all four instruments".
It is not: Spearman is +0.03, -0.09, -0.40, -0.97. The claim is now stated as
measured, and the proper-score side (+0.71 to +0.86 on all four) carries the
weight.

The script also computed Pearson while the paper said "rank correlation"; it
now reports both and the paper quotes Spearman.

### Also fixed

Removed the unsupported reachability/safety argument from App. E. The
co-improvement of S_0 and proper scores under prompt search is retained as the
measurement; the claim that the degenerate optimum was "not reachable" is not,
because the dispersion of visited candidates was never measured.

### Outstanding after Round 2

- MAJOR: variance-decomposition "individuation" -- one observation per
  person-item cell cannot separate interaction from noise.
- MAJOR: C2ST presented as a fidelity ordering rather than classifier-specific
  detectability.
- MAJOR: abstract says "strictly dominated" where Prop. 1 has an equality case.
- MAJOR: audited system still uncited; n=512 tensors not released.
- MINOR: Limitations is one long paragraph; Eq. 3 definitions; global baseline
  fitted on all respondents; "matched exactly" overstates 0.911/0.921 vs 0.910.
