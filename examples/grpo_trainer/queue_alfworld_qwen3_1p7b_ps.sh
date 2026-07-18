#!/bin/bash
# =============================================================
# 本地 2×5090 排队脚本: 等当前训练 (Qwen3-1.7B GRPO baseline 或
# HRG 臂 C 等任何 main_ppo) 结束后, 自动点火 Qwen3-1.7B PS-GRPO 臂
# (schema 协议, λ=0.1 constant, 与 4B/8B PS 臂同参)
#
# 用法 (baseline 运行期间即可后台挂起):
#   nohup bash examples/grpo_trainer/queue_alfworld_qwen3_1p7b_ps.sh \
#     > logs/qwen3_1p7b_ps_queue.log 2>&1 &
#
# 说明同 queue_alfworld_qwen3_4b_ps.sh:
#   - mai[n]_ppo 方括号防 pgrep 自匹配;
#   - sleep 180 等 Ray 完全退场;
#   - EXPERIMENT 变量同时驱动 experiment_name 与 default_local_dir,
#     与 baseline 的 ckpt 目录天然隔离 (4B 2026-07-18 踩坑教训)。
# =============================================================
set -x

mkdir -p logs

while pgrep -f "python3 -m verl.trainer.mai[n]_ppo" > /dev/null; do
    sleep 60
done

sleep 180

EXPERIMENT=qwen3_1p7b_ps_grpo bash examples/grpo_trainer/run_alfworld_qwen3_1p7b_2gpu.sh vllm \
    env.alfworld.prediction.enable=True \
    algorithm.pred_reward.enable=True \
    > logs/qwen3_1p7b_ps_grpo_$(date +%Y%m%d_%H%M%S).log 2>&1
