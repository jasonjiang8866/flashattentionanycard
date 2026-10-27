#!/usr/bin/env bash
# A Pod's container restart must not silently launch another paid experiment.
set -euo pipefail
export BO_OUTPUT_DIR="${BO_OUTPUT_DIR:-/workspace/bo-kernel-tuner/results/current}"
if [[ -e "$BO_OUTPUT_DIR/status.json" || -e "$BO_OUTPUT_DIR/manifest.json" ]]; then
  echo "Existing run at $BO_OUTPUT_DIR. Inspect it; use run.sh --resume explicitly."
else
  child_pid=''
  trap 'if [[ -n "$child_pid" ]]; then kill -TERM "$child_pid" 2>/dev/null || true; wait "$child_pid" || true; fi; exit 143' TERM INT
  bash /opt/bo-kernel-tuner/runpod/run.sh &
  child_pid=$!
  code=0
  wait "$child_pid" || code=$?
  child_pid=''
  echo "Experiment exited with code $code. Results: $BO_OUTPUT_DIR"
fi
echo "Pod remains available for result retrieval. Stop it in RunPod when finished."
exec sleep infinity
