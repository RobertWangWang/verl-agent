#!/bin/bash
# =============================================================
# Qwen3-8B + ALFWorld GRPO — 8x RTX Pro 6000D (96GB) 服务器版
# 提案 v0.2 E3: 8B 全参主结果线 (裁决者对照 @ 更大规模)
#
# 显存核算 (16 字节/参数 全参训练状态):
#   8B 状态 131GB / 8 卡 = 16.4GB/卡 + bf16 分片 ~2GB + vLLM 权重 16.4GB
#   + 激活 (micro 4, grad ckpt) ≈ 45-55GB / 96GB —— 无需 offload,余量充足
#
# 注意:
#   - RTX Pro 6000D 为带宽削减版 (~1.1TB/s vs 1.8),gen 阶段会比 4B 机器上
#     等比例更慢 —— 首 2-3 step 观察 timing_s/gen 后再决定是否调
#     gpu_memory_utilization (KV 更大) 或 micro batch;
#   - 与 4B 脚本同规模 (16 任务 x 组 8 = 128 轨迹/step),保证跨规模可比;
#   - 命令务必单行; 第一个参数是 engine,其余透传 Hydra。
#
# 用法:
#   nohup bash examples/grpo_trainer/run_alfworld_qwen3_8b_8gpu.sh \
#     > logs/qwen3_8b_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   PS 臂: 见 queue_alfworld_qwen3_8b_ps.sh (baseline 后自动接棒)
# =============================================================
set -x
ENGINE=${1:-vllm}
if [ $# -gt 0 ]; then shift; fi

num_cpus_per_env_worker=0.1

train_data_size=16
val_data_size=64
group_size=8

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=1536 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.alfworld.require_think_tags=False \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='qwen3_8b_grpo_baseline' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
