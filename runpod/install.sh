#!/usr/bin/env bash
# Shared by bootstrap and Docker: install into the explicitly selected Python.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${BO_PYTHON:?Set BO_PYTHON to the isolated environment Python}"
"$python_bin" -m pip install --no-cache-dir 'torch==2.9.0+cu128' \
  --index-url https://download.pytorch.org/whl/cu128
"$python_bin" -m pip install --no-cache-dir -r "$project_dir/runpod/requirements.txt"
"$python_bin" -m pip install --no-deps --no-cache-dir -e "$project_dir"
"$python_bin" -m pip check
