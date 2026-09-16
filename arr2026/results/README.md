# Belief-readout artifacts

Each `h4_<tag>/` holds one belief measurement:

- `summary.json` — headline statistics and the validity verdict.
- `variance_ladder.csv` — per-cluster components.
- `readout_cluster_<k>/belief_probs.npy` — the full belief simplex,
  shaped `(n_personas, 120 items, 5 response levels)`. Persona order is
  `df[df.clusters==k].iloc[:40]` on the released corpus, so rows are
  reproducible without any additional index.
- `readout_cluster_<k>/{train,test}_case_ids.csv` — the respondent ids.

Most models and interventions reported in the paper ship their tensors here.
Some do not, and App. A of the paper names them: the refreshed belief tensors,
the per-respondent score arrays, the n=512 forecast tensors, and the
cross-architecture replication tensors. Those analyses ship their printed
outputs, not their inputs, so they cannot be recomputed from this directory
alone.

**Runs marked `"valid": false` are failures, kept for the record and excluded
from every reported figure.** `h4_qwen36_27b` is one: the FP8 checkpoint
produced all-NaN logits (coverage 0.0), so it appears in no table.

Answers are on the model's raw scale. The corpus stores 55 of the 120 items
reverse-recoded, so any comparison against human data must first apply
`arr2026/scripts/hyp/_orientation.py`.
