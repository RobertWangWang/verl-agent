#!/bin/bash
# =============================================================
# HiddenRule-Gym + Qwen3-1.7B GRPO 最小闭环冒烟 (HRG-d)
# 硬件: 2x RTX 5090 (与 run_alfworld_mini.sh 同显存包络)
#
# 目的 (docs/hiddenrule_gym_design.md §5 HRG-d):
#   1. rollout -> reward -> update 在 HiddenRule 上全链路跑通 (1-2 step 即可 Ctrl+C)
#   2. 难度校准: 未训练 LLM 的 episode/success_rate 应显著高于随机策略 (0/32),
#      但远低于 50% —— 落在可学习的动态范围内
#   3. 观察 Qwen3 的 entropy (ALFWorld 上初始仅 0.19,组内多样性风险)
#
# 用法 (单行执行; 第一个参数是 engine,其余透传 Hydra):
#   bash examples/grpo_trainer/run_hiddenrule_mini.sh
# =============================================================
set -x
ENGINE=${1:-vllm}
if [ $# -gt 0 ]; then shift; fi

num_cpus_per_env_worker=0.1

train_data_size=8
val_data_size=16
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
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=hiddenrule/HiddenRuleEnv \
    env.hiddenrule.require_think_tags=False \
    env.seed=0 \
    env.max_steps=30 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_hiddenrule' \
    trainer.experiment_name='hrg_d_qwen3_1.7b_smoke' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=100 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False $@
