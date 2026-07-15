#!/bin/bash
# =============================================================
# Qwen3-4B + ALFWorld GRPO — 8x RTX Pro 6000 (96GB) 服务器版
# 提案 v0.2 P0/P1: 吞吐验收 + 裁决者对照表第一行 (GRPO baseline @ 目标规模)
#
# 与 run_alfworld_full_32gb.sh (2x32GB) 的关键差异:
#   - 4B fp32 优化器状态 64GB / 8 卡 = 8GB/卡 → **关闭全部 offload** (update 快数倍)
#   - micro batch 放大 (每卡 96GB 余量充足)
#   - gpu_memory_utilization 0.6 (KV cache 充足,rollout 提速)
#   - 实验规模先沿用 pilot 的 16x8=128 轨迹 (内存包络已验证 ~160GB 主机内存);
#     吞吐验收通过后再考虑 train_data_size=32 (256 轨迹,需主机内存 >=256GB)
#
# 注意 (从 2x5090 pilot 继承的教训, research_logs/2026-07-14_ps_grpo_s3_baseline.md):
#   - TextWorld env worker 有 ~1MB/step/worker 内存泄漏 → 服务器要么配大 swap,
#     要么沿用"每 50 步 checkpoint 后重启" 预案
#   - 命令务必单行执行; 第一个参数是 engine,其余透传 Hydra
#
# 用法:
#   吞吐验收 (跑 2-3 step 看 timing_s/step 即可 Ctrl+C):
#     bash examples/grpo_trainer/run_alfworld_qwen3_4b_8gpu.sh
#   PS-GRPO 臂 (S4 完成后):
#     bash examples/grpo_trainer/run_alfworld_qwen3_4b_8gpu.sh vllm \
#       env.alfworld.prediction.enable=True algorithm.pred_reward.enable=True \
#       trainer.experiment_name=qwen3_4b_ps_grpo
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
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
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
    trainer.experiment_name='qwen3_4b_grpo_baseline' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
