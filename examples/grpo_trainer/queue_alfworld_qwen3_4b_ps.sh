#!/bin/bash
# =============================================================
# 服务器排队脚本: 等当前训练 (Qwen3-4B GRPO baseline) 结束后,
# 自动点火 Qwen3-4B PS-GRPO 臂 (schema 协议, λ=0.1 constant)
#
# 用法 (服务器上后台挂起,baseline 运行期间即可执行):
#   nohup bash examples/grpo_trainer/queue_alfworld_qwen3_4b_ps.sh \
#     > logs/qwen3_4b_ps_queue.log 2>&1 &
#
# 实现说明 (踩坑记录, research_logs/2026-07-17 参照):
#   - pgrep 模式里的 mai[n]_ppo 方括号防止匹配到本脚本自身的命令行
#     (本文件内含 main_ppo 字样,不做处理 watcher 会永远等不到进程消失);
#   - sleep 180 等 Ray 集群完全退场,避免端口/资源竞争;
#   - gpu_memory_utilization 提到 0.85: baseline 实测每卡仅用 ~11GB/96GB,
#     且 gen 占 step 耗时 85% —— 更大 KV cache 直接提速 rollout;
#   - 若 baseline 中途崩溃,本脚本同样会点火 PS 臂 (两臂互不依赖)。
# =============================================================
set -x

mkdir -p logs

# 等待当前 main_ppo 进程消失
while pgrep -f "python3 -m verl.trainer.mai[n]_ppo" > /dev/null; do
    sleep 60
done

# Ray 退场缓冲
sleep 180

bash examples/grpo_trainer/run_alfworld_qwen3_4b_8gpu.sh vllm \
    env.alfworld.prediction.enable=True \
    algorithm.pred_reward.enable=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    trainer.experiment_name=qwen3_4b_ps_grpo \
    > logs/qwen3_4b_ps_grpo_$(date +%Y%m%d_%H%M%S).log 2>&1
