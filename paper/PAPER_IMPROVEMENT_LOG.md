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
