#!/bin/bash
# =============================================================
# verl-agent + ALFWorld GRPO 完整 baseline (S3)
# 依据 docs/grpo_baseline_guide.md 的超参重建 (原脚本 2026-07-13 被误删):
#   train 16 任务 × group 8 = 128 轨迹/step, val 128, max_steps 50,
#   lr 1e-6, kl_loss_coef 0.01, 150 epochs, checkpoint 每 50 步
# 显存配置沿用 run_alfworld_mini.sh 已验证的组合:
#   fp32 actor 双卡 FSDP 分片 + param/optimizer offload, TP=1,
#   vLLM gpu_memory_utilization=0.5, enforce_eager
#   (mini 实测峰值 24.6G allocated / 39.2G reserved 每卡)
#
# 用法 (第一个参数是 engine, 之后全部透传给 Hydra, 务必单行执行):
#   baseline:  bash examples/grpo_trainer/run_alfworld_full_32gb.sh
#   PS-GRPO:   bash examples/grpo_trainer/run_alfworld_full_32gb.sh vllm \
#                env.alfworld.prediction.enable=True \
#                algorithm.pred_reward.enable=True \
#                trainer.experiment_name=ps_grpo_lambda0.1
# =============================================================
set -x
ENGINE=${1:-vllm}
# 消费掉第一个位置参数 (engine)，让末尾的 $@ 只透传 Hydra 覆盖项
if [ $# -gt 0 ]; then shift; fi

num_cpus_per_env_worker=0.1

train_data_size=16
# val 128→64: 每个 ALFWorld env worker 常驻 ~0.44GB 系统内存，
# 128 train + 128 val = 256 workers ≈ 113GB，叠加 FSDP CPU offload (~44GB)
# 后在 step 2 撞上 Ray 的 95% 内存阈值被 OOM 杀 (2026-07-13 首跑实录)。
# 64 个验证 episode 的成功率标准误 ~±6%，跟踪训练曲线足够；
# 终评可另跑完整验证集。
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
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
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
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='grpo_baseline_32gb' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
