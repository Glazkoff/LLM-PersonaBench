# Resolve a python3 with the scientific stack on cHARISMa.
# Module names differ between login/compute nodes (login-02 has no Python/Anaconda),
# so probe explicit interpreters instead of trusting `module load`.
resolve_py() {
  local need="$1"   # e.g. "pandas,numpy,sklearn,scipy"
  local cands=()
  module load python/miniconda 2>/dev/null || true
  module load Python/Anaconda 2>/dev/null || true
  cands+=("$(command -v python3 || true)")
  cands+=(/opt/software/python/envs/google_colab_gpu_2025/bin/python3
          /opt/software/python/envs/google_colab_gpu_2024/bin/python3
          /opt/software/python/envs/colab_gpu_py310/bin/python3)
  for p in "${cands[@]}"; do
    [ -n "$p" ] && [ -x "$p" ] || continue
    if "$p" -c "import ${need//,/, }" 2>/dev/null; then echo "$p"; return 0; fi
  done
  return 1
}
