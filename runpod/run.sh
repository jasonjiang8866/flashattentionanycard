#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${BO_PYTHON:-${BO_VENV_DIR:-${BO_WORKSPACE:-/workspace/bo-kernel-tuner}/venv}/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "Environment missing. Run: bash runpod/bootstrap.sh smoke" >&2
  echo "Or set BO_PYTHON to the Python containing the installed project." >&2
  exit 2
fi
cd "$project_dir"
export PYTHONUNBUFFERED=1
exec "$python_bin" "$project_dir/runpod/launch.py" "$@"
