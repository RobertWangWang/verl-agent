#!/bin/bash
# =============================================================
# Qwen3-4B + WebShop GRPO — 8 卡服务器版 (W01–W04 泛化臂)
#
# 环境经 HTTP 薄客户端连 E 机集中式服务 (env.webshop.server_url):
#   - 本机零 WebShop 依赖 (无 gym / web_agent_site / JVM / Lucene / Ray env actor)
#   - 规避容器 pids.max 限制 (客户端每槽位只是一个线程 + HTTP 会话)
#   - 组语义 (reset 同 idx 同目标) 由服务端 WebAgentTextEnv.reset(session=idx) 保证
#
# 配置对齐 ALFWorld 比较臂 (8/32/4 = train 8 x group 4 = 32 traj/step, val 32),
# 保证跨环境可比。差异仅环境相关: max_steps=15 (WebShop 轨迹短),
# max_prompt_length=4096 (搜索结果页长)。
#
# 用法 (单行执行; 第一个参数是 engine,其余透传 Hydra):
#   W01 基线 (small 池):
#     EXPERIMENT=webshop_qwen3_4b_base_small WEBSHOP_URL=https://<tunnel>:8443 \
#       bash examples/grpo_trainer/run_webshop_qwen3_4b_8gpu.sh vllm
#   W02 std 臂 (small): 追加 env.webshop 无关的 PS 覆盖按 ALFWorld 臂配方
#   W03/W04 (all 池): 追加 env.webshop.use_small=False
# =============================================================
set -x
ENGINE=${1:-vllm}
if [ $# -gt 0 ]; then shift; fi

export VLLM_ATTENTION_BACKEND=XFORMERS
# 多核服务器 (208 核) 线程包络加固: torch/BLAS 池按核数放大会瞬时打穿容器
# pids.max (D 机实测启动峰值 20333/20480) — 配合下方 ray_init.num_cpus=32
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

EXPERIMENT=${EXPERIMENT:-webshop_qwen3_4b_base_small}
WEBSHOP_URL=${WEBSHOP_URL:?set WEBSHOP_URL to the env-service tunnel, e.g. https://u765343-ac1f-eed61445.westc.seetacloud.com:8443}

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
    data.max_prompt_length=4096 \
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
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    ray_init.num_cpus=32 \
    env.env_name=Webshop \
    env.webshop.server_url=$WEBSHOP_URL \
    env.webshop.require_think_tags=False \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name=$EXPERIMENT \
    trainer.default_local_dir=checkpoints/verl_agent_webshop/$EXPERIMENT \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
