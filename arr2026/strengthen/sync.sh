#!/bin/bash
# Push CODE ONLY to the clusters. Never touches results trees.
#
# Both cluster checkouts carry ~1.1 GB of UNTRACKED result artifacts under
# arr2026/results_euler and arr2026/results_hse. git on the cluster is on a
# different remote (upstream Zaplavnov/main) and must not be used to sync --
# a checkout or clean there would destroy those results. rsync with explicit
# excludes is the only safe channel.
set -euo pipefail
SRC="/Users/glazkov/Development/personality-twins-arr/LLM-PersonaBench/"
EULER="airi-h200:/home/glazkov/personality-twins-arr/LLM-PersonaBench/"
HSE="hse:/home/lsavchenko/personality-arr/LLM-PersonaBench/"

COMMON=(
  -avz --no-perms --no-owner --no-group
  --exclude '.git/'
  --exclude 'arr2026/results_euler/'
  --exclude 'arr2026/results_hse/'
  # arr2026/results/ was NOT excluded until a design-workflow audit caught it.
  # The local tree holds a truncated copy of at least one committed artifact
  # (h17_swiss-ai_Apertus-8B-Instruct-2509_ipip_test2/train_answers.npy is
  # 2.5 MB against a healthy sibling's 3.7 MB), and rsync would have pushed
  # that prefix over Euler's intact copy on the next code sync. Results move
  # cluster -> laptop only, and never the other way.
  --exclude 'arr2026/results/'
  --exclude 'data/raw/'                 # 150 MB, already on both
  --exclude 'data/openpsychometrics/'   # already on both
  --exclude '*.pyc' --exclude '__pycache__/'
  --exclude 'paper/main.pdf' --exclude 'paper/*.aux' --exclude 'paper/*.log'
  --exclude '.DS_Store'
)

target="${1:-both}"
case "$target" in
  euler) rsync "${COMMON[@]}" "$SRC" "$EULER" ;;
  hse)   rsync "${COMMON[@]}" "$SRC" "$HSE" ;;
  both)  rsync "${COMMON[@]}" "$SRC" "$EULER"; rsync "${COMMON[@]}" "$SRC" "$HSE" ;;
  *) echo "usage: $0 [euler|hse|both]" >&2; exit 2 ;;
esac
echo "sync ok: $target"
