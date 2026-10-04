#!/bin/bash
#SBATCH --job-name=hbs-aamas-env-fix
#SBATCH --partition=infer
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
set -euo pipefail
CAMPAIGN="$1"
export PIP_DISABLE_PIP_VERSION_CHECK=1
"$CAMPAIGN/env/bin/python" -m pip install -r "$CAMPAIGN/requirements-main.txt"
"$CAMPAIGN/native-env/bin/python" -m pip install "$CAMPAIGN/gigaevo-f72ab08" -c "$CAMPAIGN/constraints-native.txt"
"$CAMPAIGN/env/bin/python" -m pip freeze > "$CAMPAIGN/environment-main.lock"
"$CAMPAIGN/native-env/bin/python" -m pip freeze > "$CAMPAIGN/environment-native.lock"
"$CAMPAIGN/env/bin/python" -c 'import gepa, numpy, pandas, scipy, sklearn, openai, transformers, yaml; from importlib.metadata import version; assert version("gepa")=="0.1.4"; print("MAIN_IMPORTS_OK")'
"$CAMPAIGN/native-env/bin/python" -c 'import gigaevo, hydra, plotly, pandas; assert gigaevo.__version__=="2.20.1"; print("NATIVE_IMPORTS_OK")'
printf '%s\n' 'environment setup passed; not scientific evidence' > "$CAMPAIGN/ENV_READY"
