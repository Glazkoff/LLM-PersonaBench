# LLM-PersonaBench

LLM-PersonaBench is a research codebase for evaluating how well large language
models can simulate human personality profiles on the IPIP-NEO-120 questionnaire.
The pipeline builds a personality-conditioned prompt from real participant
traits and facets, asks an LLM to answer the 120 IPIP-NEO items, parses the
answers, recomputes OCEAN + 30 facet scores, and compares the simulated profile
with the original participant.

The repository also includes an EvoPrompt-based genetic optimization loop for
improving the personality prompt components per psychometric cluster.

## What Is Measured

For each participant, the simulator compares model answers against human IPIP
answers and personality scores:

- `similarity`: mean answer-level similarity.
- `avg_diff`: mean absolute difference between model and human Likert answers.
- `pearson_corr`: Pearson correlation over questionnaire answers.
- `mae_35`: mean absolute error over 35 psychometric dimensions.
- `mean_similarity_facets`: mean similarity over the 30 IPIP-NEO facets.
- `mean_similarity_traits`: mean similarity over the 5 OCEAN traits.

## Repository Layout

```text
configs/
  examples/                         Small example configs
  experiments/                      Full experiment configs used in runs
data/
  IPIP-NEO/120/questions.json       IPIP-NEO-120 question text
  IPIP-NEO/300/questions.json       IPIP-NEO-300 question text
  raw/df_ipipneo_120_clusters       Main clustered participant table, gitignored
external/evoprompt/                 EvoPrompt submodule
notebooks/
  stat_sign_metrics.ipynb           Bootstrap significance analysis
results_experiments/                Saved experiment artifacts and summary tables
src/
  simulator/person_type_opt.py      Main experiment loop
  utils/personality_match.py        Metrics, parsing status, batch evaluation
  utils/five_factor.py              OCEAN/facet recomputation via five-factor-e
  models/providers/                 Cloud API and OpenRouter model wrappers
tools/
  launch_experiment.py              Main CLI entry point
  run_model.py                      Small model smoke-test utility
```

## Installation

Use Python 3.11 or newer. The project was developed on Windows, but the code is
plain Python and should also run on Linux/macOS.

```bash
git clone --recurse-submodules <repo-url>
cd LLM-PersonaBench

python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## API Keys

Create a `.env` file in the repository root. Supported providers are:

```env
# For provider: cloud
CLOUD_API_KEY=...

# For provider: openrouter
OPENROUTER_API_KEY=...
OPENROUTER_HTTP_REFERER=https://your-project-page.example
OPENROUTER_APP_TITLE=LLM-PersonaBench
```

The `cloud` provider uses the OpenAI-compatible endpoint
`https://foundation-models.api.cloud.ru/v1`. The `openrouter` provider uses
`https://openrouter.ai/api/v1`.

## Data

The main participant table is not versioned in the repository and must be downloaded manually.

Download the processed dataset from:
https://osf.io/dsx56/overview?view_only=f236a415ffa94a6095c249070877aeac

After downloading, place the file at:

```text
data/raw/df_ipipneo_120_clusters
```

## Running Experiments

All experiment runs use:

```bash
python tools/launch_experiment.py --config <path-to-yaml>
```

The path may be absolute, relative to the repository root, or relative to
`configs/`.

### Baseline Without Optimization

This evaluates the base cluster prompt on the test split only. It is cheaper and
is the recommended first smoke test with a small `num_participants` value.

```bash
python tools/launch_experiment.py --config configs/experiments/no_opt_mean_value/gigachat3_0_cluster_no_opt_mean_value.yaml
```

For a quick dry run, copy a config from `configs/examples/`, reduce
`data.num_participants`, and set `simulation.participant_batch_size` to a value
that respects your API rate limits.

### EvoPrompt / Genetic Optimization

The full optimization pipeline first evaluates the base prompt on the test
split, then optimizes prompt components on the train split, then evaluates the
best evolved prompt on the same held-out test split.

```bash
python tools/launch_experiment.py --config configs/experiments/evoprompt_iter2/qwen3/qwen3_cluster_0_1_2_3.yaml
```

The default split is deterministic by row order within each cluster:

```text
first 60% of selected participants -> train
remaining 40% -> test
```

## Output Artifacts

Each run creates:

```text
results_experiments/<experiment_id>/
  config.json
  experiment_log.json
  result_log.json
  cluster_<id>/
    dataset_split_ids.json
    train_case_ids.csv
    test_case_ids.csv
    before_optimization_test_answers.csv
    before_optimization_test_participants.jsonl
    after_optimization_test_answers.csv
    after_optimization_test_participants.jsonl
    evolution_history.json
```

Important files:

- `result_log.json`: main summary artifact with per-cluster stage metrics.
- `before_optimization_test_answers.csv`: model answers before prompt evolution.
- `after_optimization_test_answers.csv`: model answers after prompt evolution.
- `*_participants.jsonl`: participant-level metrics and parse statuses.
- `evolution_history.json`: generation-level candidate scores and best prompts.
- `dataset_split_ids.json`: exact train/test participant IDs used for the run.

## Reproducing the Main Results

1. Install `data/raw/df_ipipneo_120_clusters` as described above.
2. Install dependencies and configure API keys.
3. Run the experiment configs from `configs/experiments/`.

For the main EvoPrompt iteration, use the configs under:

```text
configs/experiments/evoprompt_iter2/
```

```bash
python tools/launch_experiment.py --config configs/experiments/evoprompt_iter2/qwen3/qwen3_cluster_0_1_2_3.yaml
```

## Quick Model Smoke Test

To verify credentials and provider configuration without running an experiment:

```bash
python tools/run_model.py configs/examples/cloud_qwen.yaml
```

This sends a single prompt and prints the model response.
