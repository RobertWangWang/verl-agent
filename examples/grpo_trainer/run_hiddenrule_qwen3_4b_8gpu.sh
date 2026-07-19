#!/bin/bash
# =============================================================
# HiddenRule-Gym + Qwen3-4B — 8 卡服务器版 (R20+: HRG@4B, experiment_plan §2.2)
#
# 回答 pilot 遗留头号问题: 1.7B 三臂全线地板是稀疏信号问题还是模型容量问题?
#   - 臂 A@4B (本脚本默认, prediction 关): 若仍地板 → 稀疏失效结论与规模无关;
#     若学动 → 臂 A 结论加规模限定, HRG 上的 PS 问题在 4B 重新打开;
#   - PS 臂@4B: ⚠️ 等 R36/R37 定出修复配方后再跑 (λ=0.1 constant 已被
#     R06/R10 证明会触发可预测性劫持, 勿原样复用), 见文件尾配方注释。
#
# 与 1.7B pilot 的可比性 (环境侧逐项一致): HRG 默认难度 (5 房 6 机关 arity2),
#   max_steps=30 (随机地板 7.7%), 32 轨迹/step (8×4), seed 0, 150 epochs。
# 与 4B ALFWorld 线的一致性: 模型/优化器/无 offload/micro batch 全同
#   run_alfworld_qwen3_4b_8gpu.sh。
#
# ckpt 目录由 EXPERIMENT 变量派生 (与 1.7B 脚本同款防呆)。
# 服务器端惯例: 模型路径如需本地化, 手改 MODEL_PATH 一处即可。
#
# 用法 (务必单行; 第一个参数 engine, 其余透传 Hydra):
#   臂 A baseline:
#     nohup bash examples/grpo_trainer/run_hiddenrule_qwen3_4b_8gpu.sh > logs/hrg_4b_grpo_arm_a_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# =============================================================
set -x
ENGINE=${1:-vllm}
if [ $# -gt 0 ]; then shift; fi

EXPERIMENT=${EXPERIMENT:-hrg_4b_grpo_arm_a}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
CKPT_ROOT=${CKPT_ROOT:-checkpoints/verl_agent_hiddenrule}

num_cpus_per_env_worker=0.1

# 与 1.7B pilot 同规模 (32 轨迹/step); HRG episode 短 (30 步), 8 卡下步速快
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
    actor_rollout_ref.model.path=$MODEL_PATH \
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
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
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
    trainer.experiment_name=$EXPERIMENT \
    trainer.default_local_dir=$CKPT_ROOT/$EXPERIMENT \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@

# ---------------------------------------------------------------
# PS 臂@4B 配方 (⚠️ 等 R36/R37 判定后选其一, 勿用 λ=0.1 constant 原方):
#   退火版 (若 R36 有效):
#     EXPERIMENT=hrg_4b_ps_anneal bash ...本脚本... vllm \
#       env.hiddenrule.prediction.enable=True algorithm.pred_reward.enable=True \
#       algorithm.pred_reward.anneal.style=cosine
#   mean-norm 版 (若 R37 有效):
#     EXPERIMENT=hrg_4b_ps_meannorm bash ...本脚本... vllm \
#       env.hiddenrule.prediction.enable=True algorithm.pred_reward.enable=True \
#       algorithm.norm_adv_by_std_in_grpo=False
# ---------------------------------------------------------------
