#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_dir="${BO_WORKSPACE:-/workspace/bo-kernel-tuner}"
venv_dir="${BO_VENV_DIR:-$workspace_dir/venv}"
if [[ -e "$venv_dir" && ! -f "$venv_dir/.bo-kernel-managed" ]]; then
  echo "Refusing to modify an existing unmanaged environment: $venv_dir" >&2
  exit 2
fi
mkdir -p "$venv_dir"
touch "$venv_dir/.bo-kernel-managed"
python3 -m venv "$venv_dir"
export BO_PYTHON="$venv_dir/bin/python"
export BO_WORKSPACE="$workspace_dir"
bash "$project_dir/runpod/install.sh"
exec bash "$project_dir/runpod/run.sh" "$@"
