#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export PYDANTIC_AI_NO_BANNER=1
RUN_ID=confirmatory-5400-v1
TARGET=5400
LOG=results/${RUN_ID}.console.log
while true; do
  n=$(wc -l < "results/${RUN_ID}/trials.jsonl" 2>/dev/null || echo 0)
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) trials=${n}" | tee -a "$LOG"
  if [ "$n" -ge "$TARGET" ]; then
    echo "reached target" | tee -a "$LOG"
    .venv/bin/python scripts/postprocess_confirmatory.py "results/${RUN_ID}" | tee -a "$LOG"
    # mark meta complete if needed
    break
  fi
  .venv/bin/python -u scripts/run_benchmark.py --all --all-seeds --resume --run-id "$RUN_ID" >>"$LOG" 2>&1 || {
    echo "run exited nonzero; sleeping and retrying" | tee -a "$LOG"
    sleep 5
  }
  sleep 2
done
