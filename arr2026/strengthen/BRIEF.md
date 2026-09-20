# Strengthening campaign — grounding brief (2026-09-18)

Authoritative facts, verified this session. Agents MUST read this before designing or
writing code. Do not re-derive; do verify anything you depend on.

## 0. Goal

Turn the current audit paper ("A Constant Vector Beats Your Digital Twin") into a paper
whose headline is a POSITIVE, useful result: a simulator that predicts unseen human
behaviour from limited observations while preserving differences between people.

The decisive empirical question (Experiment 1 in the review):

> After correcting the population distribution, does the LLM provide predictive
> information about an unseen person that a strong conditional statistical model
> with the SAME human observations does not already capture?

Success criteria (from the review, treat as pre-registered):
- Better conditional prediction than the strongest baseline with the SAME human
  information and supervision.
- Correct-person advantage over shuffled-person controls.
- Transfer to an untouched external task.
- No material deterioration in joint or subgroup fidelity.
- A useful effect size / human-data saving, with uncertainty.
- Planning gate: >= 5% relative reduction in primary predictive loss vs. strongest
  same-information baseline, paired 95% CI excluding zero.

Primary metric: ordinal RPS (== the discrete CRPS already implemented) and conditional
log loss. Joint metrics (energy score, C2ST) are SUPPORTING checks only, never the
selection objective.

## 1. Repository

Local (authoritative working copy): `/Users/glazkov/Development/personality-twins-arr/LLM-PersonaBench`
Branch `arr-october`, HEAD `fba2c4d`. Remote `origin` = github.com/Glazkoff/LLM-PersonaBench.

Layout:
- `arr2026/scripts/hyp/` — all hypothesis scripts (h1..h23, `_corpora.py`, `_scoring.py`,
  `_orientation.py`, `_orientation_detect.py`)
- `arr2026/slurm/euler/`, `arr2026/slurm/hse/` — sbatch files
- `arr2026/results/` — committed result summaries (Mac-side)
- `arr2026/results_euler/`, `arr2026/results_hse/` — cluster-side artifact trees
- `paper/main.tex` (90 KB), `paper/refs.bib`, `paper/figures/`
- `idea-stage/` — the earlier amplification pilot (BUGGY, see §5)

## 2. Data (all local and on both clusters)

`arr2026/scripts/hyp/_corpora.py` gives ONE loader with a uniform contract:

    load(name) -> Corpus(Y, ids, text, scale, rev, n_input, recoded, K)

| name    | source                                   | respondents | items | K | n_input | recoded |
|---------|------------------------------------------|-------------|-------|---|---------|---------|
| ipip    | data/raw/df_ipipneo_120_clusters (150 MB)| ~410k       | 120   | 5 | 2/facet | True    |
| sd3     | data/PAlign/Dark-Triad.csv               | ~20k        | 27    | 5 | 4/trait | False   |
| big5    | data/openpsychometrics/BIG5              | ~19k        | 50    | 5 | 5/trait | detected|
| hexaco  | data/openpsychometrics/HEXACO            | ~22k        | ~240  | 7 | 5/facet | detected|

ORIENTATION IS A REAL TRAP. IPIP is stored already reverse-recoded; SD3 is raw. The
model always answers the LITERAL item, so on a recoded corpus the model's simplex must
be reversed on reverse-keyed targets before scoring (`P[:, flip, :] = P[:, flip, ::-1]`).
Getting this wrong is invisible in any variance statistic. HEXACO has no published key
in this distribution — its `rev` is the detector's output; do not present HEXACO
orientation agreement as confirmation of anything.

openpsychometrics BIG5/HEXACO `data.csv` also carry demographics (age, gender, country,
`race`, `engnat`) — use for subgroup calibration checks.

## 3. Saved artifacts — the calibration screen needs NO new inference

Every readout script saves, per run directory:
- `belief_probs.npy`  (n_panel, n_target_items, K)  — the model's own token distribution
  over the digits, renormalised; NaN where digit mass < 0.1
- `target_answers.npy` (n_panel, n_target_items)
- `train_answers.npy`  (n_train, n_target_items)
- `summary.json` — includes `target_ids`, `reverse_keyed_targets`, `corpus_recoded`,
  `mass_on_scale_mean`, `coverage`, `valid`
- h15 only, additionally: `code_test.npy`, `code_train.npy` — the (n, n_scales)
  percentile-style persona scores from the INPUT items

On Euler `~/personality-twins-arr/LLM-PersonaBench/arr2026/results_euler/` (1.1 GB, ~90 dirs):
- `h15_{qwen25_7b,granite_8b}_{ipip,sd3}` — the disjoint protocol, n_test=256
- `h16_{val,test2}_<model>_ipip` — 12 models, two disjoint panels
- `h17_<model>_<corpus>_{val,test2}` — 4 corpora x several models x 2 panels
- `h4*`, `h4n160_*`, `h4r_*` — same-item conditioning (WEAKER protocol, do not use for
  prediction claims; usable for population-calibration fitting only)

h17 dirs do NOT save `code_test/code_train`. If a statistical decoder needs the persona
scores, recompute them with `h17_readout.py:scores_from` + the same `panels()` split and
the same `--seed` — or add the save and re-run the (cheap) readout.

## 4. Current headline numbers (recorded, not re-derived this session)

`arr2026/results/h15/disjoint_replication.txt`, normalized CRPS, lower is better:

| run                  | correct | permuted | prior  | decoder |
|----------------------|---------|----------|--------|---------|
| h15_qwen25_7b_ipip   | 0.2510  | 0.2547   | 0.1535 | 0.1136  |
| h15_granite_8b_ipip  | 0.3578  | 0.3596   | 0.1535 | 0.1136  |
| h15_qwen25_7b_sd3    | 0.2952  | 0.3056   | 0.1687 | 0.1309  |
| h15_granite_8b_sd3   | 0.2707  | 0.2872   | 0.1687 | 0.1309  |

So: the conditional decoder (ridge on one-hot persona-score bins -> ordinal CDF,
isotonic-projected) beats the empirical prior, which beats every prompted LLM by a wide
margin. Identity gain (permuted - correct) is positive but ~20x smaller than the gap to
the prior. **Beating the raw LLM is therefore NOT a success criterion. The decoder is a
mandatory comparator and the bar to clear.**

`arr2026/results/h14/conditional_crps.txt` has the eleven-model decomposition
(correct / permuted / prior / decoder / spread D_Q / mixture excess / residual) — but it
conditions on the SAME items it predicts, so its effect sizes cannot establish prediction
of unseen behaviour. Use h15/h16/h17 (disjoint) for every predictive claim.

## 5. Known defects in prior work — do not repeat, do not cite as support

1. `idea-stage/pilot_two_knob.py:89`
   `score = vb - 2.0 * max(0.0, cr - base_crps) if (base_crps := None) else vb`
   The walrus assigns `None`, which is falsy, so the ternary ALWAYS takes `vb`. The
   advertised CRPS constraint is never enforced. Its selected (alpha, T) is therefore an
   unconstrained individuation maximiser. Its "recoverable" verdict is void.
2. The same script's quantity labelled CRPS is a squared distance between the
   population-mixture CDF and the empirical human CDF — it is unchanged by permuting
   identities, so it is not a person-conditioned forecast score.
3. That script applies no reverse-orientation transformation before comparing saved
   probabilities with the recoded IPIP corpus.
4. `idea-stage/exp_alpha_c2st_summary.json`: amplification moved mean C2ST AUC only
   0.9925 -> 0.9842 against a finite-sample human reference of 0.625. More spread is not
   fidelity.

## 6. Clusters — both live, verified 2026-09-18

### Euler (`ssh airi-h200`, 135.106.168.26) — PRIMARY
- 8x H200 NVL, 768 CPU cores (688 idle at check time), partitions `train` and `infer`,
  TIMELIMIT infinite. `/` 14T with 5.0T free.
- Project: `/home/glazkov/personality-twins-arr/LLM-PersonaBench`
- Python: `/home/glazkov/personality-twins-arr/vllmenv/bin/python`
  (numpy 2.3.5, pandas 3.0.5, torch 2.13.0+cu130, sklearn 1.9.0, scipy 1.18.1)
- Logs: `/home/glazkov/personality-twins-arr/logs`
- `export HF_HOME=/home/glazkov/hf_cache`
- CPU-only jobs MUST omit `--gres` so they never hold a GPU.
- Others share this queue (pchelin's `vulneval-*` on `train`, glazkov's `medevo-*`/`hfr-*`
  on `infer`). Size jobs honestly; long `--time` on a GPU job blocks CPU backfill.
- ALWAYS set `--time=HH:MM:SS`. Never `UNLIMITED`.

### HSE cHARISMa (`ssh hse`) — SECONDARY, use for the GPU fan-out
- Partition `rocky` (NOT `normal`, which is dead), `--account=proj_1759`, `--qos=normal`
  (the `rocky` QOS caps 1 cpu/1 gpu per user — unusable for a pipeline).
- v100/a100/h100/h200 nodes. No `--mem` (memory is not a schedulable TRES here).
- Project: `/home/lsavchenko/personality-arr/LLM-PersonaBench` (HEAD b19bd53 — BEHIND,
  must be updated before use). Also `/home/lsavchenko/personality-twins-arr/LLM-PersonaBench`.
- venvs: `~/personality-arr/{arrenv,cpuenv,hfenv,torchenv}`. `module load Python/Anaconda`
  gives a bare python3 with no pandas — always use an explicit venv python.
- `export TORCHDYNAMO_DISABLE=1` (compute nodes have GLIBC < 2.34; torch.compile crashes).
- NEVER put `type_e` in a constraint (SIGILL at python startup on those a100 hosts).
- mmBERT-class 256k-vocab models OOM at batch 8 on V100-32GB — for any HF model, size
  the batch to the node.

### Rules for both
- Write multi-line sbatch scripts LOCALLY and `scp` them. Never `ssh "cat > f <<EOF"`
  (backslash double-escaping silently corrupts python args).
- Every job: explicit `--time`, explicit `--output`/`--error` under the cluster's logs dir,
  `set -uo pipefail`, and a final `echo "DONE rc=$?"`.
- Prefer `scontrol`-free dependency chains: cancel+resubmit rather than
  `scontrol update jobid=X Dependency=...` (it can silently corrupt Features).

## 7. Literature that must be compared against or cited

| Work | Why it constrains us |
|---|---|
| Quantifying the Persona Effect (ACL 2024) | "persona variables carry signal" is not novel |
| SubPOP (arXiv 2502.16761, ACL 2025) | distributional fine-tuning on surveys, unseen-survey generalisation |
| Distribution Shift Alignment (arXiv 2510.21977) | two-stage distribution adaptation; training empirical distribution baseline |
| Beyond the Mean (arXiv 2606.28963) | structural/marginal/individual fidelity; adaptation from small human pilots |
| Calibrated Value Personas (arXiv 2605.16193) | mean-preserving dispersion adjustment == temperature repair |
| SimBench (ICLR 2026) | external group-level evaluation across 20 datasets |
| BehaviorBench / Be.FM-1.5 (arXiv 2606.24162) | individual + distributional eval, specialist checkpoints |
| Centaur (Nature 2025, Psych-101) | required comparator IF we move to cognitive decision tasks |
| OdysSim / OSim-8B (arXiv 2606.14199) | if we expand to interactive user simulation |
| Park et al. (arXiv 2411.10109) | self-report grounding predicting individuals across outcomes |

"We add calibration" or "we fine-tune with LoRA" does not carry novelty. The contribution
must be demonstrated transfer + preserved behavioural structure + reduced need for human
observations.

## 8. Protocol invariants — violate any of these and the result is void

- Split RESPONDENTS before fitting anything: clustering, quantization, retrieval,
  calibration, decoder, or training.
- For an evaluated person, all observed inputs must precede the held-out targets. Never
  use a cluster label computed from their full target-inclusive questionnaire.
- Keep paraphrases and near-duplicate items in the SAME split.
- Separate the zero-shot transfer track (NO target-label calibration) from the
  target-adapted track. Compare methods at the same target-label budget.
- Always carry: permuted-person control, no-history control, and a same-history no-LLM
  baseline.
- Report labelled humans, observed answers per person, external training data, model size,
  inference tokens, and tuning trials for every arm.
- Respondent-level paired uncertainty. Where a claim generalises across tasks, quantify
  task-level variation too.
- Include w_LLM = 0 as an admissible selected outcome in any stacking arm. Otherwise a
  successful statistical correction gets misreported as an LLM contribution.

## 9. Addendum — verified 2026-09-18 during setup

### Sync channel
Both cluster checkouts are on the UPSTREAM remote (`Zaplavnov/LLM-PersonaBench`, branch
`main`) and their entire `arr2026/` tree — including ~1.1 GB of results — is UNTRACKED.
`git checkout`/`git clean` on a cluster would destroy the result artifacts.
**The only safe sync channel is `arr2026/strengthen/sync.sh [euler|hse|both]`**, which
rsyncs code with explicit `--exclude` of `arr2026/results_euler/`, `arr2026/results_hse/`,
`data/raw/` and `data/openpsychometrics/`, and never passes `--delete`.
Euler was synced successfully at setup time.

### Python environments — corrected
| Host | Interpreter | numpy | pandas | scipy | sklearn | torch | transformers |
|---|---|---|---|---|---|---|---|
| Euler | `/home/glazkov/personality-twins-arr/vllmenv/bin/python` | 2.3.5 | 3.0.5 | 1.18.1 | 1.9.0 | 2.13.0+cu130 | (present) |
| HSE | `/home/lsavchenko/personality-arr/arrenv/bin/python` | yes | yes | **NO** | **NO** | cu128 | >=5.16 |
| HSE | `torchenv`, `cpuenv`, `hfenv` | stale/broken (`torchenv` has no numpy) | | | | | |

Consequence: **all CPU/statistical analysis runs on Euler** (full stack, 688 idle cores).
HSE's `arrenv` is a GPU-readout environment only. If an HSE job needs scipy/sklearn,
install into `arrenv` from the LOGIN node first (`uv pip install --python <arrenv> ...`)
— compute nodes have no internet, and `HF_HUB_OFFLINE=1` is set in HSE jobs.

HSE python resolution helper: `arr2026/slurm/hse/_pyresolve.sh` (`resolve_py "pandas,numpy"`).
`module load Python/Anaconda` on login-02 yields a python3 WITHOUT pandas — never rely on it.

### Local development environment
The Mac reproduces the published h15 numbers exactly with
`uv run --with numpy --with pandas --with scipy --with scikit-learn python <script>`.
Verified: `h15_analyse.py` on `arr2026/results/h15/h15_qwen25_7b_{ipip,sd3}` returns
0.2510/0.1535/0.1136 and 0.2952/0.1687/0.1309, matching `disjoint_replication.txt`.
**Smoke-test every Wave-1 CPU script locally before submitting it to Slurm.**

### Queue etiquette confirmed at setup
Euler `infer` and `train` are shared with glazkov's `medevo-*`/`hfr-*` and pchelin's
`vulneval-*`. HSE has four of lsavchenko's `arr-h4pa` jobs pending on `rocky`.

## 10. MEASURED THIS SESSION — the real bar (read before designing anything)

I measured these directly on the h15 IPIP panel (seed 260909, n_test=256, n_train=20000,
identical split to `h15_disjoint_readout.py`). Scripts:
`scratchpad/bar_probe.py`, `scratchpad/coldstart_probe.py`. Reproduced the published
incumbent exactly (0.1136 / 0.1535), so the harness is trustworthy.

### 10.1 The incumbent decoder is NOT the strongest same-information baseline

| conditional model on the SAME 60 observed answers | CRPS | vs incumbent, paired 95% CI |
|---|---:|---|
| empirical prior                                    | 0.1535 | -0.0398 [-0.0434,-0.0365] |
| **incumbent** (30 facet scores, 5 bins, one-hot)   | 0.1136 | — |
| continuous facet scores (no binning)               | 0.1163 | -0.0026 [-0.0035,-0.0018] |
| **raw 60 answers, one-hot (300 feats), ridge**     | **0.1087** | **+0.0049 [+0.0042,+0.0057]** |
| raw 60 answers, linear                             | 0.1148 | -0.0011 [-0.0021,-0.0001] |

**The published 0.1136 is beaten by a plain ridge on the raw observed answers.** Any
design that treats 0.1136 as the bar is measuring against a strawman. The bar is
<= 0.1087 and a nonlinear model (GBM / per-item ordinal logistic / low-rank item factor)
will push it lower. Every Track A arm MUST include the raw-answer decoder.

### 10.2 The item cold-start bar — the LLM's only real niche, and it is narrow

The winning decoder is fitted per TARGET ITEM on 20000 human answers to that exact item.
On a genuinely unseen item no such labels exist. Leave-one-item-out within facet (item j
unseen; only its facet sibling supervises):

| arm | CRPS |
|---|---:|
| prior, WARM (item j's own labels)   | 0.1535 |
| decoder, WARM (item j's own labels) | 0.1087 |
| prior, COLD (sibling only)          | 0.1714 |
| **decoder, COLD (sibling only)**    | **0.1422** |
| best prompted LLM on this panel (Qwen2.5-7B) | 0.2510 |
| Granite-4.1-8B on this panel        | 0.3578 |

cold-minus-warm penalty +0.0335 [+0.0317,+0.0353].

**Consequences that must shape every design:**

1. The prompted LLM at 0.2510 is not close to ANY statistical arm — not the warm decoder
   (0.1087), not the cold decoder (0.1422), not even the cold prior (0.1714). A
   calibration that merely improves the LLM is not a result.
2. Within IPIP, facet siblings are so informative that item cold start costs only 0.034.
   So "unseen item" inside the same instrument is NOT a regime where statistics fail.
   The regime where the surrogate genuinely disappears is an unseen CONSTRUCT / unseen
   INSTRUMENT — train on IPIP, predict HEXACO's honesty-humility or SD3's
   Machiavellianism, where no sibling item exists in the training corpus. Track B must
   be built around that, and the cold-start control above must be reported so the
   reader can see the surrogate being removed.
3. The realistic positive outcomes, in descending order of strength:
   (a) stacking gives a small but paired-significant gain over the strongest
       same-information statistical model — report the effect size honestly;
   (b) the LLM wins specifically in the unseen-construct regime where no label-free
       statistical surrogate exists;
   (c) the LLM reduces the number of labelled humans needed to reach a fixed error
       (the savings curve).
   Design so that a NULL on (a) is still publishable, because on these numbers a null
   on (a) is the likeliest single outcome.
4. Do not let any design report "beats the raw LLM" or "beats 0.1136" as success.

## 11. MEASURED THIS SESSION — the gate answer, and where the positive result lives

Pilot, saved probabilities only, no inference. Qwen2.5-7B on IPIP, the h15 panel of 256
scored respondents split into 128 CALIBRATION / 128 EVAL. Every alpha, temperature and
stacking weight is chosen on the calibration half and applied frozen to the eval half;
statistical baselines are fitted on the 20000 training respondents, disjoint from both.
Scripts: `arr2026/strengthen/gate_pilot.py`, `gate_cold.py`, `gate_cold_fair.py`.

### 11.1 In the warm regime the answer is NO, and it is unambiguous

| arm | CRPS | delta vs decoder, paired 95% CI |
|---|---:|---|
| population prior | 0.1538 | -0.0434 [-0.0490,-0.0383] |
| decoder on raw observed answers | 0.1103 | — |
| LLM raw, reoriented | 0.2481 | -0.1378 [-0.1447,-0.1309] |
| LLM population-recentered, alpha*=0.1 | 0.1534 | -0.0431 [-0.0487,-0.0381] |
| stack(decoder, calibrated LLM), w*=0.00 | 0.1103 | +0.0000 |

The validation-selected LLM weight is **exactly zero**, and the selected alpha collapses
to 0.1, which discards almost all of the LLM's person-specific deviation and lands the
calibrated LLM on top of the population prior. **With 20000 human answers to the target
item, the LLM is redundant.** This is the review's predicted row: "Calibration beats the
raw LLM but not the conditional decoder."

Do not try to rescue this by tuning harder. It is the correct answer for that regime, and
reporting it plainly is what makes the next result credible.

### 11.2 The label-budget effect is a TUNING ARTIFACT — corrected

A first pass held ridge lambda fixed at 1e-2 across every label budget and appeared to
show the LLM's stacking weight rising off zero as labels were removed (w*=0.60 at N=100,
gain +0.0269). **That result is void.** With 300 one-hot features against 100 labelled
respondents, the fixed-lambda decoder is overfitting, and the LLM was only repairing a
strawman.

`gate_cold_fair.py` re-runs the identical curve with the decoder tuned per budget —
lambda over 1e-2..3e3 and a shrink-to-prior coefficient, both selected on the same
calibration respondents that select the LLM weight:

| budget on target item | lambda* | shrink* | decoder | selected w* | stack | gain |
|---|---:|---:|---:|---:|---:|---:|
| N=25    | 3   | 0.50 | 0.1526 | **0.00** | 0.1526 | +0.0000 |
| N=100   | 30  | 0.25 | 0.1421 | **0.00** | 0.1421 | +0.0000 |
| N=400   | 100 | 0.00 | 0.1276 | **0.00** | 0.1276 | +0.0000 |
| N=2000  | 100 | 0.00 | 0.1171 | **0.00** | 0.1171 | +0.0000 |
| N=20000 | 100 | 0.00 | 0.1103 | **0.00** | 0.1103 | +0.0000 |

Note also that the properly tuned decoder is monotone in budget (0.1526 -> 0.1103), as it
must be; the earlier non-monotonicity (N=100 worse than N=25) was the overfitting
signature that gave the artifact away.

**On IPIP, with Qwen2.5-7B, a fairly tuned conditional decoder makes calibrated LLM
information redundant at every human-label budget from 25 to 20000.** Prediction-level
convex stacking selects zero LLM weight everywhere.

### 11.3 The standing methodological rule this establishes

At every label budget the statistical comparator must be at ITS best — capacity tuned on
validation respondents at that budget — and the tuning must be reported. Any Track A, B
or E arm that puts a baseline on a label budget at fixed hyperparameters is measuring a
strawman and its result will not survive review. This rule cost us a headline result
within an hour of finding it; it will cost a reviewer nothing to find the same thing.

### 11.4 Where the positive result can still live — the open regimes

Three regimes remain untested and are where the campaign's effort belongs. They are
ordered by how much they would be worth if they come in positive.

1. **Feature-level combination instead of prediction-level.** A scalar convex weight
   between two CDFs is a weak combiner: it can only interpolate, so an LLM carrying
   information ORTHOGONAL to the decoder's features is invisible to it. The stronger test
   is to put the LLM's own CDF for target item j in as EXTRA FEATURES alongside the 300
   one-hot observed-answer features and refit. If the LLM carries anything the observed
   answers do not, that regression will find it and a convex weight cannot. **This test
   must be run before any negative conclusion is drawn.**
2. **Genuinely unseen constructs.** In the COLD arm the decoder still borrowed the target
   item's facet sibling, and within IPIP siblings are extremely informative (cold-minus-
   warm cost only +0.034). The regime where the surrogate genuinely disappears is an
   unseen INSTRUMENT or unseen CONSTRUCT — train on IPIP, predict HEXACO honesty-humility
   or SD3 Machiavellianism, where no sibling item exists in the training corpus at all.
   Saved belief probabilities for ipip / sd3 / big5 / hexaco already exist on Euler
   (`h17_<model>_<corpus>_{val,test2}`), so this is CPU-only and needs no new inference.
3. **A much better elicitation.** 0.2481 is a very weak LLM readout — it is worse than the
   population prior. Eleven models and four corpora of saved probabilities exist; before
   concluding "LLMs carry no incremental signal", establish which model and which
   elicitation gets closest, because the current conclusion may be a statement about one
   weak persona prompt rather than about LLMs.

### 11.5 What this means for the campaign

- The gate question is answered NO for the specific combination tested (IPIP, Qwen2.5-7B,
  prediction-level convex stacking, warm and budget-limited supervision). That is a real,
  reportable result and it is the honest floor of the paper.
- Do not build the headline on the budget curve until regime 1 above is tested with a
  tuned baseline. If regime 1 is also null, the strongest remaining route is regime 2.
- Every arm must report w* (or the fitted LLM coefficients), so the reader sees the method
  concede when the LLM adds nothing. A method that knows when not to fire is a stronger
  claim than one that always fires.
- Report absolute and relative gains, and where a positive regime is found, convert it
  into the paper's headline number: the labelled humans per item the LLM is worth, with a
  confidence interval.

## 12. Recipe — reconstructing observed answers for any saved run, CPU-only

`h17_readout.py:panels()` is fully deterministic given `(eligible, seed, n_hold, n_train,
n_panel, panel)`, and `eligible` is just the respondents with a complete persona from
`scores_from(c, inp)`. So every h16/h17 run's `scored` and `train` respondent indices — and
therefore their RAW input-half answers — can be reconstructed exactly, with no GPU and no
re-inference, even though those directories save no `code_*.npy`:

```python
from _corpora import load
from h17_readout import split_items, scores_from, panels
c = load(corpus)                                  # ipip | sd3 | big5 | hexaco
inp, tgt = split_items(c.ids, c.scale, c.n_input)
S, names = scores_from(c, inp)
complete  = np.where(~np.isnan(S).any(axis=1))[0]
scored, train = panels(complete, args)            # seed 260910, n_hold 256, n_train 8000,
                                                  # n_panel 128, panel in {val,test2}
idx = {i: k for k, i in enumerate(c.ids)}
X_scored = c.Y[np.ix_(scored, [idx[i] for i in inp])]   # the observed answers
X_train  = c.Y[np.ix_(train,  [idx[i] for i in inp])]
```

Verify the reconstruction before trusting it: `c.Y[np.ix_(scored, [idx[j] for j in tgt])]`
must equal the run's saved `target_answers.npy` exactly (allow NaN). h15 used seed 260909,
n_test 256, n_train 20000 and a plain `rng.permutation(len(Y))` — a DIFFERENT split
function; do not mix the two.

Consequence: **the entire scaled Wave-1 screen — every model x {ipip, sd3, big5, hexaco}
x {val, test2}, every calibration and combination arm — is CPU-only on Euler from the
1.1 GB of artifacts already sitting in `arr2026/results_euler/`.** No GPU job is needed to
answer the gating question at full scale.

Use the exact vectorised isotonic projection in `arr2026/strengthen/iso_fast.py` rather
than `h15_analyse.py:isotonic_rows`: it is validated identical to the incumbent and to a
brute-force L2 projection on adversarial inputs (monotone, ties, out-of-range, K-1=6), and
is ~36x faster, which is what makes a hyperparameter-tuned budget curve affordable.

## 13. IPIP-NEO-300 — a real same-person unseen-item-family set (and a claim to NOT make)

`data/PAlign/Test-set.json`: 300 respondents x 300 IPIP-NEO-300 items, with `case`, `sex`,
`age`, `country` and a timestamp. `data/PAlign/IPIP-NEO-ItemKey.xls` gives `Full#` (1-300),
`Short#` (1-120), `Sign`, `Key`, `Facet` and the item text for all 300 items — i.e. the
exact 120 <-> 300 mapping and a published key for every item.

### 13.1 Do NOT claim cross-file same-person linkage — I checked, and it is false

181 of the 300 `case` ids also appear in `data/raw/df_ipipneo_120_clusters`. That overlap
is an ID COLLISION, not a linkage. On those 181 supposedly-identical respondents:

| check | value |
|---|---|
| exact cell match on the same 120 items | 0.2845 (chance for 5 levels is ~0.20) |
| mean absolute difference | 1.0808 |
| Pearson r across cells | 0.1808 |
| rows fully identical | 0 of 181 |
| sex agrees | 0.564 |
| age agrees | **0.039** |
| country agrees | 0.624 |

Age agreeing 3.9% of the time is chance. The two files number their respondents
independently. **Any "test-retest" or "repeated observation" claim built on this join is
false, and so is any reliability ceiling derived from it.** The review's request for
repeated observations cannot be met from local data; say so in Limitations.

### 13.2 What the file DOES give, and it is exactly what Track B needs

Within Test-set.json alone, every respondent answered both the 120 short-form items and
the 180 items that are NOT in the short form:

- 180 held-out items, spanning all 30 facets
- 181+ respondents with a complete held-out matrix (the full 300 for the items checked)
- conditioning instrument = the IPIP-NEO-120, which is exactly the instrument the rest of
  the paper conditions on

So this is **same people, same conditioning protocol, 180 items never used for
conditioning and never seen by any arm** — genuine unseen-item-family transfer, and a true
external evaluation set that no arm has been developed on. It is the strongest transfer
evidence available from local data, and it costs one GPU readout of 300 x 180 prompts.

It is also the natural low-label regime: with only 300 respondents, a per-item statistical
decoder has at most a few hundred labels per held-out item — precisely where §11.2 says
the LLM's weight might finally rise off zero, and where a text-conditioned model can be
fitted on the 120 items and transferred to the 180.

Protocol requirements: split the 300 respondents before fitting anything; keep the
short-form items strictly as conditioning; report the per-facet breakdown, since all 30
facets are represented; use the published `Sign` key for orientation rather than the
detector.

## 14. Feature-level combination is ALSO null — the warm regime is closed

§11.4 flagged prediction-level convex stacking as too weak a combiner to detect an LLM
signal orthogonal to the decoder's features. `arr2026/strengthen/gate_feature.py` runs the
stronger test: the LLM's own CDF for target item j enters as EXTRA FEATURES beside the
decoder prediction and the population prior, and a combiner is fitted on the 128
calibration respondents (128 x n_items pooled rows) and frozen on the 128 eval
respondents. The control is the identical combiner, identical capacity, identical fitting
rows, WITHOUT the LLM features.

| model / corpus | combiner | no LLM | with LLM | gain, paired 95% CI |
|---|---|---:|---:|---|
| Qwen2.5-7B / IPIP    | linear | 0.1101 | 0.1102 | -0.0000 [-0.0001,+0.0001] |
| Qwen2.5-7B / IPIP    | GBM    | 0.1125 | 0.1124 | +0.0000 [-0.0005,+0.0005] |
| Granite-4.1-8B / IPIP| linear | 0.1101 | 0.1102 | -0.0000 [-0.0001,+0.0000] |
| Granite-4.1-8B / IPIP| GBM    | 0.1125 | 0.1123 | +0.0002 [-0.0003,+0.0006] |
| Qwen2.5-7B / SD3     | linear | 0.1235 | 0.1237 | -0.0002 [-0.0005,+0.0000] |
| Qwen2.5-7B / SD3     | GBM    | 0.1324 | 0.1334 | -0.0009 [-0.0025,+0.0007] |
| Granite-4.1-8B / SD3 | linear | 0.1235 | 0.1237 | -0.0002 [-0.0004,-0.0000] |
| Granite-4.1-8B / SD3 | GBM    | 0.1324 | 0.1344 | -0.0019 [-0.0039,-0.0000] |

**Three independent combiner families — convex weight, linear feature stack, gradient-
boosted feature stack — across two models and two instruments all return zero or negative.**
In the warm regime the LLM's answer distribution carries no information about a person that
their own observed answers do not already carry. This is now a well-controlled negative,
not a single failed attempt, and it should be reported as such: it is the finding that
motivates the method, exactly as the review says the metric analysis should.

The warm regime is closed. All remaining campaign effort belongs in the two regimes where
a statistical decoder cannot be fitted at all:
  (a) held-out CONSTRUCTS — whole facets/traits with no labelled items (§11.4 regime 2);
  (b) the IPIP-NEO-300 held-out 180 items at ~300 respondents (§13.2), the lowest-label
      regime available and the only genuine same-person unseen-item-family set.

## 15. THE HEADLINE — the answer-equivalent, m* = 1

`arr2026/scripts/hyp/h26_answer_equivalent.py`. "The LLM adds nothing" is true and
useless. This converts it into one interpretable number with a confidence interval.

A persona prompt is built from all 60 of a respondent's observed IPIP answers (as 30
facet scores). So price it: **a decoder given m of that person's actual answers matches
the calibrated LLM persona at m = ?** The ridge penalty is retuned at every m, so no
budget is beaten by an overfitted baseline.

IPIP, h15 panel, EVAL half, normalized ordinal RPS:

| m observed answers | 0 | 1 | 2 | 3 | 5 | 8 | 12 | 20 | 30 | 45 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| decoder loss | .1538 | .1509 | .1493 | .1466 | .1408 | .1367 | .1319 | .1256 | .1199 | .1142 | .1103 |

| run | LLM raw | LLM calibrated | alpha* | m* (integer crossing) | **m* interpolated** |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B / IPIP     | 0.2481 | 0.1534 | 0.1 | 1 of 60 | **0.12** |
| Granite-4.1-8B / IPIP | 0.3555 | 0.1536 | 0.1 | 1 of 60 | **0.04** |
| Qwen2.5-7B / SD3      | 0.2962 | 0.1665 | 0.3 | 1 of 12 | **0.07** |
| Granite-4.1-8B / SD3  | 0.2660 | 0.1628 | 0.6 | 1 of 12 | **0.27** |

Report BOTH. The integer crossing is what a practitioner acts on ("ask one more
question"). But it is coarse and it OVERSTATES the LLM: on IPIP the calibrated persona
sits at 0.1534, between m=0 (0.1538) and m=1 (0.1509) and far nearer the floor. Placing
it on the curve between the bracketing budgets gives the honest effect size.

**A persona prompt built from sixty of a person's real answers is worth between 0.04 and
0.27 of one answer — roughly a tenth of a single question.** The curve is smooth and
monotone in m, the signature of a correctly tuned baseline at every budget.

### Why this is the paper's positive contribution, not a failure report

The contribution is a MEASUREMENT INSTRUMENT plus the verdict it returns:

- The answer-equivalent is model-agnostic, instrument-agnostic and
  elicitation-agnostic. It prices any simulator in the currency a practitioner
  actually spends — questions asked of a real person — and it is directly comparable
  across the literature in a way a CRPS delta is not.
- It is meaningful in both directions. m* = 40 would be strong evidence that persona
  prompting substitutes for a questionnaire; m* = 1 says a practitioner should ask one
  more question instead. The instrument does not presuppose the answer, which is exactly
  why the m* = 1 result is credible.
- It is paired with a combination framework that reports w* and concedes when the LLM is
  redundant. A method that knows when not to fire is a stronger claim than one that
  always fires.
- Every future improvement — better elicitation, fine-tuning, a text-conditioned model —
  is measured on the same scale, so the paper supplies the field's yardstick rather than
  one more point estimate.

### What must be established before this ships

1. **Breadth.** m* must be measured across all 191 saved cells (H25/H26 on Euler), not
   three. If any model or corpus gives m* well above 1, that cell is the positive result
   and the paper leads with it.
2. **Elicitation is not the confound.** 0.2481 raw is worse than the population prior, so
   a reviewer will say the persona prompt, not the LLM, is what was measured. The screen
   across every model and every elicitation in the artifact set is the answer; report the
   BEST cell's m*, not the mean.
3. **Transfer.** m* on the IPIP-NEO-300 unseen-item-family set (H24), where per-item
   supervision is ~150 respondents rather than 20000 — the regime most favourable to the
   LLM.
4. **Uncertainty.** m* is a threshold crossing on a noisy curve; report it with a
   bootstrap over respondents AND over observed-item subsets, and report the loss gap at
   m* with its CI rather than m* alone.

## 16. BREADTH (68 cells, Euler h26) — the positive result, and it replicates

The three-cell m*~0.1 figure in §15 was partly a statement about a weak model. Priced
across 68 saved cells (18 ipip, 18 sd3, 16 big5, 16 hexaco; every model in the artifact
set; both disjoint panels) the picture is different and much better.

### 16.1 The measurement is stable

Spearman correlation of m* between the `val` and `test2` panels, over the 32 (model,
corpus) pairs scored on both: **rho = +0.777, p = 1.7e-07**. The answer-equivalent is a
property of the model and instrument, not of a panel draw. That is what licenses using it
as an instrument at all.

### 16.2 There is a model ladder, and the best persona is worth ~3 answers

IPIP, mean over both panels:

| model | raw LLM loss | alpha* | **m* interpolated (of 60)** |
|---|---:|---:|---:|
| Qwen3.6-35B-A3B          | 0.172 | 0.85 | **3.07** |
| Qwen3.8-27B              | 0.183 | 1.10 | **1.31** |
| granite-4.2-30b          | 0.243 | 0.45 | 0.45 |
| gemma-4-12B-it           | 0.267 | 0.35 | 0.37 |
| gpt-oss-20b              | 0.188 | 0.70 | 0.21 |
| granite-4.2-8b           | 0.376 | 0.45 | 0.20 |
| GLM-4.7-Flash            | 0.243 | 0.20 | 0.18 |
| Qwen2.5-7B-Instruct      | 0.248 | 0.10 | 0.13 |
| granite-4.1-8b           | 0.356 | 0.10 | 0.04 |
| Apertus-8B-Instruct-2509 | 0.286 | 0.00 | 0.00 |

Overall across 68 cells: median m* 0.145, mean 0.377, **max 3.24**.

### 16.3 The mechanism: value lives in what survives calibration, not in accuracy

| predictor of m* | Spearman rho | p |
|---|---:|---:|
| calibration coefficient alpha* | **+0.771** | 1.5e-14 |
| raw LLM loss | -0.430 | 2.6e-04 |
| calibrated LLM loss | -0.174 | 0.16 (ns) |

alpha* is how much of the LLM's person-specific deviation from its own population mean
survives validation selection. It predicts the answer-equivalent almost perfectly, while
the calibrated loss does not predict it at all. **A persona's worth is not its accuracy;
it is the size of the person-specific component that survives population correction.**
Apertus-8B has alpha*=0 -- validation throws its deviations away entirely -- and is worth
exactly zero answers. That is a clean, mechanistic, and testable statement, and it is what
the paper should explain.

### 16.4 What is and is not claimable

- CLAIMABLE: the instrument; its val/test2 stability; the ladder; the alpha* mechanism;
  "the best current persona is worth about three of sixty answers on IPIP".
- NOT CLAIMABLE from a single cell: the top cell's gap CI excludes zero on `val`
  (+0.00049, +0.00688) but NOT on `test2` (-0.00265, +0.00326). Lead with the ladder and
  the rank correlation, which are robust, and report the per-cell CIs honestly.
- HEXACO is the weakest corpus (median m* 0.000) and big5 next (0.065); ipip 0.295 and
  sd3 0.402 are the strongest. Report per-corpus rather than pooling.

### 16.5 Immediate consequence for the GPU queue

Qwen2.5-7B and Granite-4.1-8B, the two models the h24 transfer readout was originally
pointed at, sit at the BOTTOM of the ladder (m* 0.13 and 0.04). Running the unseen-item
transfer test on them would measure nothing. The transfer readout must target
**Qwen3.6-35B-A3B and Qwen3.8-27B**, the two models with a measurable answer-equivalent,
plus one weak model as a contrast.

## 17. THE DECISIVE SCREEN — 178 cells, no incremental LLM value anywhere

`h25_incremental_screen.py` over every saved run on Euler. 191 candidate directories,
178 scored (12 h16 runs were correctly REFUSED for a split mismatch and re-run after the
fix in §17.3; 1 invalid at mass 0.0). Four corpora: big5 54, ipip 42, sd3 42, hexaco 40.
Roughly thirteen models, both disjoint panels.

| combiner | cells with a significant positive increment | mean gain | max gain |
|---|---:|---:|---:|
| convex stack | 27/178 | +0.00001 | +0.00010 |
| linear feature stack | **0/178** | -0.00008 | +0.00021 |
| gradient-boosted feature stack | 8/178 | -0.00011 | +0.00141 |

Selected stacking weight w*: **0.00 in 145 of 178 cells, 0.05 in the other 33. Never
above 0.05.**

### 17.1 Read the "significant" counts correctly

534 tests were run (178 cells x 3 combiners). At a 5% one-sided rate, chance alone
produces ~27. The convex stack's 27 and the GBM's 8 are at or below that, and the largest
convex-stack gain across every cell is +0.0001 — three orders of magnitude below the
0.11-0.15 loss scale. **Report these with a Holm-Bonferroni or BH correction across the
cell grid and state the multiplicity explicitly.** Uncorrected counts in this table must
not appear in the paper.

### 17.2 This does not contradict the answer-equivalent ladder — it completes it

The two measurements price the LLM against different opponents, and both are needed:

- Against a decoder holding FULL per-item human supervision (20000 labelled respondents),
  the LLM adds nothing: 0/178 on the strongest linear combiner, w* <= 0.05 everywhere.
- Priced in the currency of the person's OWN answers (§16), the best persona is worth
  about 3 of 60, and that worth rises monotonically with model capability.

Together these bound the contribution precisely: a persona carries a small, real,
model-scaling amount of person-specific information, and that amount is completely
dominated by — and redundant given — a handful of the person's actual answers. That is a
sharp, quantified, useful statement, and it is the paper.

### 17.3 A protocol trap worth a Limitations sentence

The h16 `val`/`test2` runs were produced by `h15_disjoint_readout.py`, so they carry the
h15 summary schema and the h15 permutation (seed 260909, a fixed 256-respondent reserved
head). h17 uses a different seed (260910) and permutes only respondents with a complete
persona. Reconstructing h16 under h17's rule silently yields the WRONG respondents.

The screen caught this only because it verifies every reconstruction against the run's own
saved `target_answers.npy` and refuses to score a mismatch. Without that check, 12 runs
would have been scored against other people's answers and the error would have been
invisible in every summary statistic. **Keep the verify-or-refuse rule in every script
that reconstructs a split, and say so in the paper.**

## 18. HARDENING PASS — defects the design workflow found in this session's own code

The design workflow's audits were run against the campaign code written earlier in the
same session and found five real defects. All are fixed; the h15 numbers are unchanged by
the fixes, which is the correct outcome — the guards protect dirty cells without moving
clean ones.

| # | Defect | Consequence | Fix |
|---|---|---|---|
| 1 | `h25:rps` built indicators as `(Y <= t)`; `NaN <= t` is False at every threshold, which is exactly the indicator pattern for "answered K" | a MISSING human answer was scored as a top-category answer — a fabricated label | mask non-finite answers to NaN; cell is dropped |
| 2 | partially-NaN forecast rows were averaged over their non-NaN thresholds | a model REFUSING to answer was rewarded; `_scoring.py:51` uses `nansum` and scores an all-NaN forecast a perfect 0.0 | any NaN in a forecast row makes the whole cell NaN |
| 3 | arms were each reduced over their own valid cells | LLM arms (which refuse) were compared against statistical arms (which never do) on DIFFERENT item sets | one arm-independent common-support mask, with the dropped-cell count printed and recorded per run |
| 4 | stale pre-key-fix run directories scored alongside corrected ones | big5 carried 19 rows per panel where other corpora carried 12 | admissibility filter on the corpus-specific expected reverse-key count (ipip 43, sd3 3, big5 7, hexaco 46). Filter on the KEY COUNT, never on directory spelling — ipip/sd3/hexaco all use the "p" spelling a spelling filter would delete |
| 5 | `sync.sh` excluded `results_euler/` and `results_hse/` but NOT `arr2026/results/` | the next code push would have rsynced the laptop's tree over Euler's | `--exclude 'arr2026/results/'` added; dry run verified to touch zero result paths |

### 18.1 One genuinely corrupt artifact, found and repaired

A scan of all 322 `.npy` files under `arr2026/results*` found exactly one unreadable:
`arr2026/results/h17/h17_swiss-ai_Apertus-8B-Instruct-2509_ipip_test2/train_answers.npy`,
truncated at 331,760 of 480,000 elements, **on both the laptop and Euler**, and tracked in
git. It was not a sync artifact; the source file is short.

Repaired safely rather than regenerated: all six healthy `*_ipip_test2` siblings have
byte-identical `train_answers.npy` (one SHA over six runs), because h17 runs on the same
corpus and panel share a training block. The truncated file's readable prefix was verified
element-for-element against a sibling before copying, proving it is a pure truncation of
the same data. Both copies now load, shape (8000, 60), fully finite.

### 18.2 Multiplicity, now controlled

The screen runs one one-sided test per (cell, combiner). Across the grid that is **506
tests; 43 are nominally significant at 5% — exactly the chance rate — and ZERO survive
Benjamini-Hochberg at q<0.05.** The uncorrected counts in §17 must never be reported
without this line. `make_strengthen_assets.py` now computes it and writes
`paper/multiplicity.txt`.

### 18.3 Still outstanding

- `h17n512` `test2` has already been scored end-to-end by h25/h26, so it is development
  data, not a holdout. Any confirmatory claim needs a panel no analysis has touched.
- No equivalence margin was pre-registered, so "the LLM adds nothing" is reported as
  "nothing detectable at this sample size", not as demonstrated equivalence. Stated in
  Limitations.
- `_scoring.py`'s `nansum` bug is untouched — this campaign does not use it, but every
  h1..h23 script that does should be re-checked before its numbers are re-quoted.

## 19. HARDENED RE-RUN (2026-09-20) — what changed, and the transfer result

All Euler jobs completed, exit 0:0, queue empty. `hbs-h24t` 8/8 GPU readouts,
`hbs-h25` and `hbs-h26` 4/4 shards each.

### 19.1 Status reconciliation — the fixes did what they should

| status | before | after |
|---|---:|---:|
| ok | 178 | **160** |
| reconstruction-mismatch | 12 | **0** (h16 split fix) |
| stale-key rejected | 0 | **30** (big5 pre-key-fix dirs, rev=9 vs expected 7) |
| invalid (mass 0.0) | 1 | 1 |

178 + 12 − 30 = 160. Exactly as intended. Common support turned out to be nearly
irrelevant here (min kept 99.997%, max 2 cells dropped in any run), so the earlier
numbers were not distorted by it — but the guard stays, because that is a property of
this artifact set and not of the method.

### 19.2 The boundary claim needed correcting: ONE cell now survives BH

Under the hardened scorer: **426 tests, 35 nominal, 1 survives BH q<0.05** — previously
reported as zero. It is `h17_Qwen_Qwen3p8-27B_hexaco_val`, GBM combiner, +0.00092
[+0.00046,+0.00140], q=0.026.

It does not replicate, and the paper says so rather than filtering it out:

| panel | n_eval | gbm gain |
|---|---:|---:|
| h17 val | 64 | **+0.00092** (sig) |
| h17 test2 | 64 | −0.00034 (sign reverses) |
| h17n512 val | 256 | +0.00006 |
| h17n512 test2 | 256 | −0.00007 |

A fifteen-fold shrinkage under a four-fold panel increase is a small-sample artefact.
Every other nominally significant HEXACO cell sits on the same 64-respondent val panel
and behaves identically. w* is now 0.00 in 127 of 160 cells, 0.05 in the rest, never
higher.

### 19.3 IPIP-NEO-300 transfer — the best number in the paper

8 readouts, all valid (mass >= 0.993, coverage 1.0, 93 of 180 reverse-keyed targets
correctly flagged), 150 scored / 150 train, 120 conditioning / 180 target items.

| persona | raw S | calibrated S | alpha* | m* (of 120) |
|---|---:|---:|---:|---:|
| Qwen3.6-35B-A3B, profile | 0.1172 | 0.0860 | 0.9 | **5.38** |
| Qwen3.8-27B, profile | 0.1473 | 0.0906 | 1.1 | 2.23 |
| Qwen3.6-35B, history m=120 | 0.1462 | 0.0927 | 0.3 | 1.10 |
| granite-4.2-30b, profile | 0.2261 | 0.0930 | 0.7 | 0.73 |
| Qwen3.6-35B, history m=60 | 0.1478 | 0.0935 | 0.2 | 0.49 |
| Qwen3.6-35B, history m=20 | 0.1470 | 0.0938 | 0.2 | 0.19 |
| Qwen2.5-7B, profile | 0.2166 | 0.0944 | 0.3 | 0.03 |
| Qwen3.6-35B, history m=5 | 0.1484 | 0.0944 | 0.1 | 0.00 |

Two findings worth the paper:

1. **The price is HIGHEST where human labels are scarcest.** 5.38 of 120 on the unseen
   item family against 3.24 of 60 on the short form, with the model ordering preserved.
   That is the regime a practitioner is actually in when simulation is tempting.
2. **Profile beats transcript, by ~5x.** The same model, same items, given its persona as
   the literal 120 (item, answer) pairs prices at 1.10; given the 30-line trait profile
   computed from those same answers it prices at 5.38, and retains alpha*=0.9 against the
   transcript's 0.3. The history budget curve is monotone (0.00, 0.19, 0.49, 1.10 at
   m=5,20,60,120), so this is not a budget effect — the aggregation is doing the work.
   How observations are presented matters more than how many there are.

### 19.4 Paper state

Title, abstract, intro, the three new sections, transfer subsection, Related work,
Limitations and Conclusion all updated to the hardened numbers (89 priced cells, 160
screened cells, 426 tests, 15 models, 5 inventories). Body fits exactly 8 pages;
Limitations p9; 24 pages total; zero LaTeX warnings; all citations resolve. A stale-number
sweep confirms no pre-hardening figure survives in the text.

## 20. HSE arm (2026-09-20) — small-model end of the transfer ladder

`hbs-h24h` array 4337734: **4/4 COMPLETED, exit 0:0, on cn-051 (H200)** after the
V100 retarget. The earlier array (4337541) died on every V100 with
`torch.AcceleratorError: no kernel image is available` — HSE's `arrenv` carries a cu128
torch with no sm_70 kernels. Constraint is now `type_f|type_h` (H100 cn-047/048,
H200 cn-049..051); `type_e` stays excluded for the startup SIGILL.

All four readouts are genuine: P = (150, 180, 5), mass 0.975-0.999, coverage 0.99-1.0,
valid. The short runtimes (2-8 min) are legitimate for 0.5B/2B models on an H200.

### 20.1 The transfer ladder now spans 0.5B to 35B across two clusters

| model | persona | raw S | alpha* | m* (of 120) |
|---|---|---:|---:|---:|
| Qwen3.6-35B-A3B | profile | 0.1172 | 0.9 | **5.38** |
| Qwen3.8-27B | profile | 0.1473 | 1.1 | 2.23 |
| Qwen3.6-35B-A3B | hist m=120 | 0.1462 | 0.3 | 1.10 |
| granite-4.2-30b | profile | 0.2261 | 0.7 | 0.73 |
| gemma-4-E2B-it | hist m=20 | 0.1810 | 0.1 | 0.54 |
| Qwen3.6-35B-A3B | hist m=60 | 0.1478 | 0.2 | 0.49 |
| gemma-4-E2B-it | profile | 0.2507 | 0.1 | 0.22 |
| Qwen3.6-35B-A3B | hist m=20 | 0.1470 | 0.2 | 0.19 |
| Apertus-0.5B | profile | 0.1308 | 0.8 | 0.13 |
| Apertus-0.5B | hist m=20 | 0.1260 | 0.7 | 0.06 |
| Qwen2.5-7B | profile | 0.2166 | 0.3 | 0.03 |
| Qwen3.6-35B-A3B | hist m=5 | 0.1484 | 0.1 | 0.00 |

### 20.2 CORRECTION — "profile beats transcript" is NOT universal

§19.3 stated the aggregation does the work. With the HSE small models that claim is
too strong and the paper has been qualified accordingly:

| model | profile | transcript (m=20) | direction |
|---|---:|---:|---|
| Qwen3.6-35B-A3B | 5.38 | 0.19 | profile, by ~28x |
| Apertus-0.5B | 0.13 | 0.06 | profile |
| gemma-4-E2B-it | 0.22 | 0.54 | **transcript** |

Two of three favour the profile and the strong model favours it overwhelmingly, but
gemma-4-E2B reverses it. Three models cannot settle the general direction. What the
price DOES establish is that the presentation choice is worth several observations and
must therefore be reported — which is a claim about the instrument, not about prompting.

## 21. Slurm hygiene — two findings on HSE

### 21.1 `hbs-h4path` was failing on an unconditional preflight, not on data

Jobs 4337750-53 exited 2 with `PREFLIGHT FAIL: missing $f`. All four files it checks are
present. The script (`~/personality-twins-arr/hse_h4path.sbatch`) was written through an
ssh heredoc and line 28 reached disk as

    [ -e "\$f" ] || { echo "PREFLIGHT FAIL: missing \$f"; exit 2; }

The escaped `\$f` tests for a file literally named `$f`, so the preflight can NEVER pass,
and the message cannot name the file it is complaining about. This is the exact trap in
the standing rule: write multi-line sbatch locally and `scp` it, never `ssh "cat > f <<EOF"`.
Fixed in place with `sed` (backup kept alongside) and verified by dry-running the loop:
`preflight ok`.

Still outstanding on that script, NOT changed because it alters scheduling and it is not
mine to redesign: it requests **no `--gres`** yet calls `nvidia-smi` and loads a model, so
it is using a GPU it never reserved; and it carries **no `--constraint`**, so nothing
stops it landing on a `type_e` a100 that SIGILLs at python startup.

### 21.2 The `hbs-` rename was incomplete

The first pass covered only `arr2026/slurm/` inside the repo. Most of this project's
sbatch files live in the cluster working directories ABOVE the checkout. Completed:
**49 files renamed on HSE, 45 on Euler**; zero `job-name=arr-` remain on either cluster
(95 and 90 now carry `hbs-`).
