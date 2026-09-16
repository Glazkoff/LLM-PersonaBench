# Reproduction bundle

The five input sets the paper's repository is too small to hold, plus the CPU
commands that regenerate every reported number depending on them. No GPU and no
LLM calls are needed: these are the saved model outputs, and everything below is
a re-reduction of them.

    inputs/h4_refresh/       4.2M   belief tensors, 4 current-generation models (App. L)
    inputs/h21_respondent/   596K   per-respondent S_0 / CRPS / SD arrays (App. M)
    inputs/h17_n512/         501M   n=512 forecast tensors, 12 models x 4 instruments (App. D)
    inputs/hse_repl/          13M   cross-cluster replication readouts (App. F)
    inputs/a100_repl/         13M   cross-architecture (A100) replication readouts (App. F)

`MANIFEST.sha256` lists a SHA-256 for all 568 files;
`MANIFEST.sha256.sha256` covers the manifest itself.

## 0. Integrity

    cd <bundle> && sha256sum -c MANIFEST.sha256

## 1. Setup

`$REPO` is a checkout of the paper's repository; `$B` is this unpacked bundle.
Only numpy, pandas and scipy are required.

## 2. App. L — belief variance with intervals

    python $REPO/arr2026/scripts/hyp/h22_variance_ci.py \
      --results "$B/inputs/h4_refresh" \
      --tags h4r_ibm-granite_granite-4_2-8b h4r_Qwen_Qwen3_8-27B \
             h4r_openai_gpt-oss-20b h4r_google_gemma-4-12B-it \
      --n-boot 2000

Expected: Gemma-4 VR_between 0.3342 [0.3033, 0.3453], share_between 0.5038;
gpt-oss VR_between 0.1269 [0.1181, 0.1298].

## 3. App. M — reachability stability

    python $REPO/arr2026/scripts/hyp/h23_reachability_ci.py \
      --results "$B/inputs/h21_respondent" \
      --tags h21_ibm-granite_granite-4_2-8b h21_Qwen_Qwen3_8-27B \
             h21_openai_gpt-oss-20b h21_google_gemma-4-12B-it \
      --n-boot 2000

Expected REACHABLE frequencies: Granite 0.0%, Qwen3.8 77.4%, gpt-oss 99.2%,
Gemma-4 0.0%.

## 4. App. D — nine-rule selection

    python $REPO/arr2026/scripts/hyp/h17_selection.py \
      --results "$B/inputs/h17_n512" --prefix h17n512 \
      --corpora ipip sd3 big5 hexaco --boot 4000

Expected paired differences against rps: s0 +0.298 [+0.250, +0.368] resolved;
brier +0.003 [-0.008, +0.025] unresolved; logloss +0.004 [-0.007, +0.022]
unresolved.

## 5. App. F — cross-cluster and cross-architecture

    ARR_HSE_REPL="$B/inputs/hse_repl" \
    ARR_EULER_RESULTS="$REPO/arr2026/results_euler" \
      python $REPO/arr2026/scripts/hyp/h20_crosscluster.py

    ARR_A100_REPL="$B/inputs/a100_repl" \
    ARR_EULER_RESULTS="$REPO/arr2026/results_euler" \
      python $REPO/arr2026/scripts/hyp/h20d_scores.py

Expected: cross-cluster bit-identical (0 modal flips of 76,800); cross-
architecture mean |A100 - H200| of 1.59e-03 for S_0, 3.37e-03 for mode accuracy,
4.69e-03 for S_1/2.

## What is NOT here

The eleven-model belief tensors of the main-body variance section ship in the
repository itself, under `arr2026/results*/h4_*/`. The GPU runs that produced
every tensor here are not reproducible from this bundle -- it contains their
outputs. Regenerating the tensors themselves needs the sbatch scripts under
`arr2026/slurm/` and the model weights.
