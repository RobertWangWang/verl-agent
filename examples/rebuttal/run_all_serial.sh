#!/usr/bin/env bash
# 串行运行全部已实现的 A/B 正式训练臂：17 臂 × seeds 0,42,96 = 51 runs。
# 默认只生成并检查配置；正式执行：bash examples/rebuttal/run_all_serial.sh --execute
# 每次使用全部 8 张 GPU，完成一个 run 后再开始下一个；失败即停，不覆盖已有输出。
# 后续参数可覆盖下方默认值，例如 --seeds 0 或 --output-root /mnt/data/another_run。
# Python 沿用当前环境，也可通过 PYTHON_BIN 指定。W/X 和正式终点评估不在此入口内。
set -euo pipefail

REBUTTAL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$REBUTTAL_SCRIPT_DIR/run_train.sh" \
    --arms all \
    --seeds 0,42,96 \
    --model-4b /mnt/data/qwen3-4b \
    --model-1.7b /mnt/data/qwen3-1.7b \
    --gpus 0,1,2,3,4,5,6,7 \
    --output-root /mnt/data/experiment_output \
    "$@"
