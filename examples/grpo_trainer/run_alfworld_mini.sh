#!/bin/bash
# =============================================================
# verl-agent + ALFWorld GRPO 最小闭环冒烟脚本
# 硬件目标: 双卡 RTX 3090 24G (系统内存建议 >= 32G)
# 模型: Qwen2.5-1.5B-Instruct 全参 (优化器/参数 offload 到 CPU)
#
# 为什么必须双卡: actor 以 fp32 加载(fsdp_workers.py 中 actor 角色固定 float32),
# 且 update_actor 会把参数和 Adam 状态全部搬回 GPU。单卡 world_size=1 时 FSDP
# 不产生分片收益, 峰值为:
#   fp32 参数 6.2G + fp32 梯度 6.2G + Adam m/v 12.3G + bf16 副本 3.1G
#   + vLLM 常驻权重 3.1G ≈ 31G  > 24G
# 双卡分片后每卡约 14G + 3.1G ≈ 17G, 可以装下。
# 用法: 放到 verl-agent 仓库根目录下执行
#   bash run_alfworld_mini_3090.sh
# 目的: 验证 rollout -> reward -> update 训练链路,
#       跑通 2~3 个 step 即可 Ctrl+C, 不追求分数
# =============================================================
set -x
ENGINE=${1:-vllm}
# 消费掉第一个位置参数 (engine)，让末尾的 $@ 只透传 Hydra 覆盖项。
# 否则 `bash run_alfworld_mini.sh vllm foo=bar` 会把 'vllm' 也传给 Hydra，
# 报 "Error parsing override 'vllm'"。
if [ $# -gt 0 ]; then shift; fi
# 不要设 XFORMERS: 这是 vLLM 0.6/0.8 时代的遗留写法(仓库其他脚本仍带着)。
# vLLM 0.11 的 V1 引擎下 XFORMERS 后端要求 paged KV block size 能被 256 整除,
# 而默认 block_size=16, 会在第一次 generate_sequences 时抛
#   RuntimeError: Paged KV cache block size must be divisible by 256
# 留空让 vLLM 自动选后端(sm_86 + flash-attn 已装 -> FlashAttention)。
# export VLLM_ATTENTION_BACKEND=XFORMERS

num_cpus_per_env_worker=0.1   # 每个环境 worker 占用的 CPU 资源

# ---- 最小化采样规模: 每个 step 采 8x4=32 条轨迹 ----
train_data_size=8
val_data_size=16
group_size=4

# 数据准备(仅用于指定模态和数据规模)
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
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=30 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='grpo_mini_smoke_3090' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=100 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False $@

# =============================================================
# 训练侧 OOM 时按顺序尝试(改一项跑一次):
# 1. actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
# 2. data.max_response_length=256
# 3. 模型换 Qwen/Qwen2.5-0.5B-Instruct
#
# 注意 gpu_memory_utilization 是 vLLM 的显存预算, 调低它并不能缓解训练侧 OOM,
# 反而会让 vLLM 分不到 KV cache 而直接报
#   ValueError: No available memory for the cache blocks.
# 该预算需覆盖 (权重 + CUDA context + 激活峰值 + KV cache), 24G 卡上 1.5B 模型
# 至少要 0.5; 若确实要给训练侧腾显存, 优先降 ppo_micro_batch_size_per_gpu。
#
# 闭环成功标志:
#   vLLM 启动 -> ALFWorld 扫描 8810 个游戏文件 -> rollout 进度条
#   -> 打印 step:1 的 reward / success rate 等 metrics -> 进入 step:2
#
# 跑通后放大规模的方向:
#   train_data_size=16, group_size=8, max_steps=50,
#   test_freq=5, val_before_train=True, logger 加回 wandb
# =============================================================