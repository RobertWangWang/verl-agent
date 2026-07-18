#!/bin/bash
# =============================================================
# Qwen3-1.7B + ALFWorld GRPO — 2x RTX 5090 (32GB) 本地版
# 规模阶梯第三点 (1.7B → 4B → 8B), 裁决者对照表跨规模行 (提案 E3)
#
# ⚠️ 配置对齐铁律: train 8 × group 4 = 32 轨迹/step, val 32 —— 与
#   4B/8B 服务器上**实际运行**的手改配置逐项一致 (4B 仓库脚本写的 16×8
#   并未被使用, 见 research_logs/2026-07-17_qwen3_4b_alfworld.md §6)。
#   三个规模若配置不一致, 跨规模对比作废。
#
# 显存: 沿用本机已验证组合 (fp32 双卡 FSDP 分片 + param/optimizer
#   offload, vLLM util 0.5, enforce_eager)。1.7B 状态 27GB offload 到
#   主机内存; 64 env workers ≈ 28GB 主机内存, 无 Ray 阈值风险。
#
# ckpt 目录由 EXPERIMENT 变量派生 —— baseline 与 PS 臂天然隔离,
#   不会出现 resume_mode=auto 误续别臂 checkpoint 的事故
#   (4B PS 臂 2026-07-18 踩坑实录)。
#
# 用法 (务必单行; 第一个参数 engine, 其余透传 Hydra):
#   baseline:
#     nohup bash examples/grpo_trainer/run_alfworld_qwen3_1p7b_2gpu.sh > logs/qwen3_1p7b_grpo_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   PS 臂: 用 queue_alfworld_qwen3_1p7b_ps.sh 自动接棒 (推荐), 或手动:
#     EXPERIMENT=qwen3_1p7b_ps_grpo nohup bash examples/grpo_trainer/run_alfworld_qwen3_1p7b_2gpu.sh vllm env.alfworld.prediction.enable=True algorithm.pred_reward.enable=True > logs/qwen3_1p7b_ps_grpo_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# =============================================================
set -x
ENGINE=${1:-vllm}
if [ $# -gt 0 ]; then shift; fi

EXPERIMENT=${EXPERIMENT:-qwen3_1p7b_grpo_baseline}

num_cpus_per_env_worker=0.1

# 与 4B/8B 实际运行配置对齐 (勿改, 见文件头)
train_data_size=8
val_data_size=32
group_size=4

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
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
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
    trainer.experiment_name=$EXPERIMENT \
    trainer.default_local_dir=checkpoints/verl_agent_alfworld/$EXPERIMENT \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
