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

# 防呆: 若在 baseline 的 main_ppo 还没启动时就挂本脚本 (数据准备阶段约 1 分钟),
# 直接进入"等消失"循环会立即通过 → 180s 后 PS 臂与 baseline 撞车 (2026-07-18 实录)。
# 先等 main_ppo 出现 (最多 30 分钟), 再等它消失。
waited=0
until pgrep -f "python3 -m verl.trainer.mai[n]_ppo" > /dev/null; do
    sleep 30
    waited=$((waited + 30))
    if [ $waited -ge 1800 ]; then
        echo "WARN: 30min 内未见 main_ppo, 视为 baseline 已结束, 继续接棒"
        break
    fi
done

while pgrep -f "python3 -m verl.trainer.mai[n]_ppo" > /dev/null; do
    sleep 60
done

sleep 180

EXPERIMENT=qwen3_1p7b_ps_grpo bash examples/grpo_trainer/run_alfworld_qwen3_1p7b_2gpu.sh vllm \
    env.alfworld.prediction.enable=True \
    algorithm.pred_reward.enable=True \
    > logs/qwen3_1p7b_ps_grpo_$(date +%Y%m%d_%H%M%S).log 2>&1
