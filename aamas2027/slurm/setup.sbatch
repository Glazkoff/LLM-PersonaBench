#!/bin/bash
#SBATCH --job-name=hbs-aamas-env
#SBATCH --partition=infer
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
set -euo pipefail
CAMPAIGN="$1"
BASEPY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
export PIP_DISABLE_PIP_VERSION_CHECK=1
"$BASEPY" -m venv --system-site-packages "$CAMPAIGN/env"
"$CAMPAIGN/env/bin/python" -m pip install 'gepa==0.1.4' 'pytest==8.4.2' 'scikit-learn==1.7.2'
mkdir -p "$CAMPAIGN/gigaevo-f72ab08"
tar -xzf "$CAMPAIGN/hbs-gigaevo-f72ab08.tar.gz" -C "$CAMPAIGN/gigaevo-f72ab08"
printf '%s\n' f72ab08c628cc49f4c4c1b6ee48bd11b68d3ce08 > "$CAMPAIGN/gigaevo-f72ab08/SOURCE_SHA"
"$BASEPY" -m venv "$CAMPAIGN/native-env"
"$CAMPAIGN/native-env/bin/python" -m pip install "$CAMPAIGN/gigaevo-f72ab08"
"$CAMPAIGN/env/bin/python" -m pip freeze > "$CAMPAIGN/environment-main.lock"
"$CAMPAIGN/native-env/bin/python" -m pip freeze > "$CAMPAIGN/environment-native.lock"
"$CAMPAIGN/env/bin/python" -c 'import gepa, numpy, pandas, scipy, sklearn, openai, transformers; from importlib.metadata import version; assert version("gepa")=="0.1.4"; print("MAIN_IMPORTS_OK")'
"$CAMPAIGN/native-env/bin/python" -c 'import gigaevo, hydra, plotly; assert gigaevo.__version__=="2.20.1"; print("NATIVE_IMPORTS_OK")'
printf '%s\n' 'environment setup passed; not scientific evidence' > "$CAMPAIGN/ENV_READY"
