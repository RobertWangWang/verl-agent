#!/bin/bash
# =============================================================
# 8B 服务器排队脚本: 等当前训练 (Qwen3-8B GRPO baseline) 结束后,
# 自动点火 Qwen3-8B PS-GRPO 臂 (schema 协议, λ=0.1 constant)
#
# 用法 (8B 服务器上后台挂起,baseline 运行期间即可执行):
#   nohup bash examples/grpo_trainer/queue_alfworld_qwen3_8b_ps.sh \
#     > logs/qwen3_8b_ps_queue.log 2>&1 &
#
# 说明同 queue_alfworld_qwen3_4b_ps.sh (mai[n]_ppo 防自匹配 / Ray 退场缓冲)。
# =============================================================
set -x

mkdir -p logs

while pgrep -f "python3 -m verl.trainer.mai[n]_ppo" > /dev/null; do
    sleep 60
done

sleep 180

bash examples/grpo_trainer/run_alfworld_qwen3_8b_8gpu.sh vllm \
    env.alfworld.prediction.enable=True \
    algorithm.pred_reward.enable=True \
    trainer.experiment_name=qwen3_8b_ps_grpo \
    > logs/qwen3_8b_ps_grpo_$(date +%Y%m%d_%H%M%S).log 2>&1
