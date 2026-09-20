#!/bin/bash
#SBATCH --job-name=hbs-env
#SBATCH --partition=rocky
#SBATCH --account=proj_1759
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/home/lsavchenko/personality-arr/logs/env_%j.out
#SBATCH --error=/home/lsavchenko/personality-arr/logs/env_%j.err
set -euo pipefail
cd /home/lsavchenko/personality-arr
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 torchenv 2>&1 | tail -1
uv pip install --python torchenv/bin/python \
  torch --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2
uv pip install --python torchenv/bin/python \
  "transformers>=4.51" "tokenizers>=0.21" "huggingface-hub>=0.30" \
  accelerate safetensors pandas numpy scipy scikit-learn 2>&1 | tail -2
torchenv/bin/python -c "import torch,transformers,pandas,numpy;print('OK torch',torch.__version__,'tf',transformers.__version__)"
echo "DONE rc=$?"
