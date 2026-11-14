#!/usr/bin/env bash
# Start API model scripts in parallel, one process each; logs go to logs/<model>.log.
#   ./run_all.sh                                  # every script in api/
#   ./run_all.sh deepgram_nova_3 gradium_stt      # a subset
# Safe to re-run: each script skips clips it already finished.
set -euo pipefail
cd "$(dirname "$0")"
AWAKE=(); command -v caffeinate >/dev/null && AWAKE=(caffeinate -i)
mkdir -p logs
models=("$@")
[ ${#models[@]} -eq 0 ] && for f in api/*.py; do models+=("$(basename "$f" .py)"); done
for m in "${models[@]}"; do
  nohup "${AWAKE[@]}" uv run "api/$m.py" >> "logs/$m.log" 2>&1 &
  echo "started $m -> logs/$m.log"
done
