#!/usr/bin/env bash
# 默认生成正式训练配置；添加 --execute 才串行运行。完整参数：bash "$0" --help
set -euo pipefail

REBUTTAL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REBUTTAL_REPO_ROOT="$(cd -- "$REBUTTAL_SCRIPT_DIR/../.." && pwd)"
REBUTTAL_PYTHON="${PYTHON_BIN:-python}"
if [[ "$REBUTTAL_PYTHON" == */* && "$REBUTTAL_PYTHON" != /* ]]; then
    REBUTTAL_PYTHON="$PWD/$REBUTTAL_PYTHON"
fi
export PYTHONPATH="$REBUTTAL_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$REBUTTAL_PYTHON" -u -m examples.rebuttal.run_matrix train "$@"
