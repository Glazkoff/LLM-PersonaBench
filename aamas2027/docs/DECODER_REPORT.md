# Fresh global-panel decoder references

`aamas2027/decoders.py` adds CPU-only, target-supervised references for the fixed
complete-case IPIP120 panel. It reads the campaign's existing `panels.json` and
corpus, checks the manifest/panel/corpus hashes and panel sizes, and uses exactly
the 60 registered input items to forecast the other 60 ordinal items. The
`train=8000` panel fits all models and the itemwise empirical prior. The
`select=120` panel chooses ridge penalties at each answer budget, an ExtraTrees
leaf size, and the full-history reference family. `opt`, `cal`, and `eval` never
take part in model fitting or decoder selection. No cluster routing or LLM calls
occur.

The families are an empirical target-item prior, a raw-answer one-hot
multioutput ridge model for the four cumulative thresholds per target, and a
multioutput ExtraTrees model (64 trees, leaf sizes 10 or 30). Ridge penalties
are selected among 1, 10, 100, and 1000. Ordinal CDFs receive bounded isotonic
projection before conversion to probabilities. Missing targets are masked
*before* the `<=` threshold comparison. Ridge fits each output on observed
labels if training targets are incomplete; the registered tree reference
requires complete-case training targets. Selection uses respondent-mean raw
normalized RPS, with a deterministic tie rule. These modest finite grids are
fixed in source; expanding them after seeing evaluation outcomes would be a new
analysis.

For retained-information curves, three seeded random permutations of the 60
input items are fixed before selection. Their prefixes are nested at
`m = 0, 1, 2, 5, 10, 20, 30, 40, 60`; ridge is tuned separately at every positive
budget on `select`. `m=0` uses the training prior. The final file stores every
respondent's RPS at every budget and order, alongside full-history prior,
ridge, ExtraTrees, and selected-family losses. The existing
`analysis.retained_information` handles crossings and censoring. The optional
`mstar_summary` bootstraps paired respondents and the registered orders and
reports fractions of interpolated, below-grid, and above-grid draws. A CI over
interpolated draws is conditional on crossing; it is not substituted for a
censored interval. The result measures retained predictive information under
this decoder, **not literal saved questionnaire answers**.

Run in a finite CPU Slurm allocation after `panels.json` is prepared:

```text
uv run --python .venv/bin/python --no-project python -m aamas2027.decoders \
  --old-root /path/to/personality-twins-arr/LLM-PersonaBench \
  --results /path/to/campaign-results \
  --manifest aamas2027/manifest.json --stage all
```

`--stage select` writes `decoders/FROZEN.json` before evaluation rows are
accessed; `--stage eval` later verifies it and writes `decoders/FINAL.json` and
`decoders/eval_probabilities.npz`. The default `all` runs those steps in that
order. Output paths refuse overwrite. `FINAL.json` includes ordered evaluation
IDs, per-person losses, curve losses, and the prediction-file hash for paired
analysis. The freeze receipt binds the manifest, panel, corpus, selected
hyperparameters, and all input orders. No model inference is rerun beyond
fitting these statistical references on the fixed training panel.

Local verification (synthetic fixtures only):

```text
UV_CACHE_DIR=/private/tmp/hbs-plan-uv-cache uv run --python .venv/bin/python \
  --no-project python -m pytest -q tests/optim/test_campaign_decoders.py \
  tests/optim/test_campaign_analysis.py
# 7 passed
```

The tests check missing-label masking, a known same-person signal against the
prior, valid projected CDFs, the nested curve and three input orders, immutable
selection under changed evaluation labels, and both interpolated and censored
`m*`. No fresh campaign score is claimed here. CPU walltime and memory for
8,000 people and the full curve need measurement in the scheduled allocation;
the 64-tree reference is deliberately finite and single-threaded per estimator
to avoid nested oversubscription.
