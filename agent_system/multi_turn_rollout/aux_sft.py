# Copyright 2026 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
S6b: 监督辅助损失臂的辅助批构造 (docs/ps_grpo_integration_design.md §8.2)

对每个带 gold_predict 的 step-sample 构造一条"教师强制"行:
prompt 不变;response = 模型自己的 response 中 <predict> 块原位替换为 gold 块
(保留 think 前缀与 action 后缀,不教偏格式)。gold 段 token 区间记入
aux_token_mask;trainer 端以 advantages = β·aux_token_mask 走标准 PPO 更新通道
—— ratio≈1 处梯度等价于加权交叉熵,zero verl-core 改动。

辅助批走独立的第二遍 compute_log_prob + update_actor:
- 不进 compute_advantage → 与 GRPO 组内对比天然隔离;
- 不进 episode 指标 → 无统计污染;
- ref_log_prob := old_log_probs → KL 项在强制 token 上梯度为 0 (低方差估计器
  在非采样 token 上有偏,置零而非引入伪拉力)。
"""

import re
from typing import Optional

import numpy as np
import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

_PREDICT_BLOCK_RE = re.compile(r'<predict>.*?</predict>', re.DOTALL | re.IGNORECASE)


def build_aux_sft_batch(batch: DataProto, tokenizer, fraction: float = 1.0,
                        seed: int = 0, placebo_shuffle: bool = False) -> Optional[DataProto]:
    """
    从训练 batch 构造辅助 SFT 批。

    要求 batch.non_tensor_batch 含 'gold_predict' (无 predict 块的行被跳过)。
    返回的 DataProto 继承父行全部 tensor/non_tensor 键 (布局零风险),其中
    responses/input_ids/attention_mask/position_ids 重建,新增 aux_token_mask。
    advantages/old_log_probs 留给 trainer 端在 compute_log_prob 后填充。
    返回 None 表示无可用行。

    placebo_shuffle (R12c 安慰剂): gold 串在候选行间随机置换 —— 破坏信号内容,
    完整保留计算量/更新次数/掩码结构。R12 > R12c 才能把增益归因于信号内容
    而非"每步多一次梯度更新"。
    """
    if 'gold_predict' not in batch.non_tensor_batch:
        return None
    golds = batch.non_tensor_batch['gold_predict']

    responses = batch.batch['responses']          # (B, R) 右 padding
    prompts = batch.batch['prompts']              # (B, P) 左 padding
    attention_mask = batch.batch['attention_mask']  # (B, P+R)
    resp_len = responses.shape[1]
    prompt_len = prompts.shape[1]
    pad_id = tokenizer.pad_token_id

    candidates = []
    for i in range(len(golds)):
        gold = golds[i]
        if not gold:
            continue
        resp_text = tokenizer.decode(responses[i], skip_special_tokens=True)
        if _PREDICT_BLOCK_RE.search(resp_text) is None:
            continue  # 无 predict 块的响应上下文不明,跳过 (parse_valid 常 >0.95)
        candidates.append(i)
    if not candidates:
        return None

    if fraction < 1.0:
        rng = np.random.RandomState(seed)
        keep = max(1, int(len(candidates) * fraction))
        candidates = sorted(rng.choice(candidates, size=keep, replace=False).tolist())

    gold_map = {i: golds[i] for i in candidates}
    if placebo_shuffle and len(candidates) > 1:
        perm_rng = np.random.RandomState(seed + 7919)
        perm = perm_rng.permutation(len(candidates))
        gold_map = {candidates[r]: golds[candidates[perm[r]]]
                    for r in range(len(candidates))}

    idx = torch.as_tensor(candidates, dtype=torch.long)
    new_responses = torch.full((len(candidates), resp_len), pad_id, dtype=responses.dtype)
    aux_token_mask = torch.zeros((len(candidates), resp_len), dtype=torch.float32)
    resp_attn = torch.zeros((len(candidates), resp_len), dtype=attention_mask.dtype)

    for row, i in enumerate(candidates):
        resp_text = tokenizer.decode(responses[i], skip_special_tokens=True)
        m = _PREDICT_BLOCK_RE.search(resp_text)
        prefix, suffix = resp_text[:m.start()], resp_text[m.end():]
        gold_block = f"<predict>{gold_map[i]}</predict>"
        # 逐段 tokenize 拼接 → gold 段 token 区间精确已知 (边界合并差异可接受,
        # 强制目标即定义本身)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        gold_ids = tokenizer.encode(gold_block, add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            suffix_ids = suffix_ids + [tokenizer.eos_token_id]
        ids = (prefix_ids + gold_ids + suffix_ids)[:resp_len]
        n = len(ids)
        new_responses[row, :n] = torch.as_tensor(ids, dtype=responses.dtype)
        resp_attn[row, :n] = 1
        g0, g1 = len(prefix_ids), min(len(prefix_ids) + len(gold_ids), resp_len)
        aux_token_mask[row, g0:g1] = 1.0

    # 白名单构造: 只带 update 所需张量。禁止整行继承 —— 父行的
    # old_log_probs/advantages/token_level_* 与旧 responses 对齐,带入即污染。
    prompt_attn = attention_mask[idx, :prompt_len]
    full_attn = torch.cat([prompt_attn, resp_attn], dim=1)
    tensors = {
        'prompts': prompts[idx].clone(),
        'responses': new_responses,
        'input_ids': torch.cat([prompts[idx], new_responses], dim=1),
        'attention_mask': full_attn,
        'position_ids': compute_position_id_with_mask(full_attn),
        'aux_token_mask': aux_token_mask,
    }
    non_tensors = {'is_aux_sft': np.ones(len(candidates), dtype=bool)}
    aux = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors,
                              meta_info=dict(batch.meta_info))
    return aux


def compute_interference_metrics(logp_pre: torch.Tensor, logp_post: torch.Tensor,
                                 response_mask: torch.Tensor) -> dict:
    """
    S6 干扰探针: aux 更新对任务批策略行为的位移 (损失通道的干扰观测,
    对应负迁移文献的梯度冲突信号; 参数空间梯度余弦需 FSDP 侵入, 此为行为学等价物)。
    - task_shift_mean > 0: aux 更新提高任务行为似然 (梯度对齐);
      < 0: 降低 (冲突 —— 干扰的直接证据);
    - task_shift_meanabs: 位移幅度 (aux 对策略的总扰动量)。
    """
    mask = response_mask.float()
    delta = (logp_post - logp_pre) * mask
    n = mask.sum().clamp(min=1.0)
    return {
        'aux_sft/task_shift_mean': (delta.sum() / n).item(),
        'aux_sft/task_shift_meanabs': (delta.abs().sum() / n).item(),
    }


def apply_aux_sft_supervision(aux: DataProto, beta: float, use_kl_loss: bool) -> DataProto:
    """
    compute_log_prob(aux) 之后调用: 把监督信号写成 PPO 语义。
    - advantages = β · aux_token_mask (强制 token 上的加权 CE, §8.2 第 3 条);
    - returns 同 advantages (dp_actor 不用,占位保持键完整);
    - use_kl_loss 时 ref_log_prob := old_log_probs (KL 梯度在强制 token 上归零)。
    """
    adv = beta * aux.batch['aux_token_mask']
    aux.batch['advantages'] = adv
    aux.batch['returns'] = adv.clone()
    if use_kl_loss and 'old_log_probs' in aux.batch.keys():
        aux.batch['ref_log_prob'] = aux.batch['old_log_probs'].clone()
    return aux
