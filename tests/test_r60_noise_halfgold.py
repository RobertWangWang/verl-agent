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

"""R60a/R60b: 正则化机制证伪臂 (prereg 2026-07-31 review-r2 batch)

R60a noise_sign: 辅助优势 β → 每 token Rademacher ±β (纯梯度噪声, 无一致目标)。
R60b half_gold: gold 内部 token 以 p 独立换为词表随机 token (<predict> 标签不动)。
"""

import numpy as np
import torch

from agent_system.multi_turn_rollout.aux_sft import (
    _corrupt_token_ids,
    apply_aux_sft_supervision,
    build_aux_sft_batch,
)
from verl import DataProto


class CharTokenizer:
    """字符级 mock tokenizer: id = ord(char); 可逆,便于精确断言"""

    pad_token_id = 0
    eos_token_id = 3
    vocab_size = 1000
    all_special_ids = [0, 3]

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        vals = [int(t) for t in ids]
        return ''.join(chr(v) for v in vals
                       if v > 3 or not skip_special_tokens and v != self.pad_token_id)


def _make_batch(resp_texts, golds, prompt_len=6, resp_len=96):
    tok = CharTokenizer()
    n = len(resp_texts)
    prompts = torch.full((n, prompt_len), tok.pad_token_id, dtype=torch.long)
    responses = torch.full((n, resp_len), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros((n, prompt_len + resp_len), dtype=torch.long)
    for i, text in enumerate(resp_texts):
        p_ids = tok.encode(f"P{i}ask:")
        prompts[i, prompt_len - len(p_ids):] = torch.as_tensor(p_ids)
        attn[i, prompt_len - len(p_ids):prompt_len] = 1
        r_ids = tok.encode(text)[:resp_len]
        responses[i, :len(r_ids)] = torch.as_tensor(r_ids)
        attn[i, prompt_len:prompt_len + len(r_ids)] = 1
    batch = DataProto.from_dict(
        tensors={'prompts': prompts, 'responses': responses, 'attention_mask': attn,
                 'position_ids': torch.zeros_like(attn),
                 'old_log_probs': torch.zeros((n, resp_len)),
                 'advantages': torch.randn(n, resp_len)},
        non_tensors={'gold_predict': np.array(golds, dtype=object)},
        meta_info={'k': 'v'},
    )
    return batch, tok


RESP = "<think>hmm</think><predict>next_location: room 9</predict><action>go</action>"
GOLD = "next_location: room 3"


def _aux_with_logp(placebo_mode=None, seed=0, **build_kw):
    batch, tok = _make_batch([RESP], [GOLD])
    aux = build_aux_sft_batch(batch, tok, seed=seed, placebo_mode=placebo_mode,
                              **build_kw)
    assert aux is not None
    aux.batch['old_log_probs'] = torch.zeros_like(aux.batch['aux_token_mask'])
    return aux, tok


class TestNoiseSign:
    def test_signs_are_rademacher_pm_beta(self):
        aux, _ = _aux_with_logp()
        aux = apply_aux_sft_supervision(aux, beta=0.1, use_kl_loss=False,
                                        noise_sign=True, noise_seed=7)
        adv = aux.batch['advantages']
        mask = aux.batch['aux_token_mask']
        on = adv[mask.bool()]
        assert on.numel() > 0
        # 掩码内只允许 ±β 两个值
        assert torch.all((on - 0.1).abs().lt(1e-6) | (on + 0.1).abs().lt(1e-6))
        # 掩码外严格为零 (噪声不得泄漏到非强制 token)
        assert torch.all(adv[~mask.bool()] == 0)

    def test_both_signs_present(self):
        # 足够长的 gold 段: 两种符号都应出现 (P(全同号) = 2^-(n-1) 可忽略)
        long_gold = "next_location: " + " ".join(f"room {i}" for i in range(12))
        batch, tok = _make_batch(
            [RESP.replace("next_location: room 9", long_gold)], [long_gold],
            resp_len=256)
        aux = build_aux_sft_batch(batch, tok)
        aux.batch['old_log_probs'] = torch.zeros_like(aux.batch['aux_token_mask'])
        aux = apply_aux_sft_supervision(aux, beta=0.1, use_kl_loss=False,
                                        noise_sign=True, noise_seed=1)
        on = aux.batch['advantages'][aux.batch['aux_token_mask'].bool()]
        assert (on > 0).any() and (on < 0).any()

    def test_seed_determinism_and_freshness(self):
        signs = []
        for noise_seed in (5, 5, 6):
            aux, _ = _aux_with_logp()
            aux = apply_aux_sft_supervision(aux, beta=0.1, use_kl_loss=False,
                                            noise_sign=True, noise_seed=noise_seed)
            signs.append(aux.batch['advantages'].clone())
        assert torch.equal(signs[0], signs[1])      # 同 seed 可复现
        assert not torch.equal(signs[0], signs[2])  # 换步 (seed) 新鲜采样

    def test_default_off_keeps_constant_beta(self):
        aux, _ = _aux_with_logp()
        aux = apply_aux_sft_supervision(aux, beta=0.1, use_kl_loss=False)
        on = aux.batch['advantages'][aux.batch['aux_token_mask'].bool()]
        assert torch.all((on - 0.1).abs() < 1e-6)

    def test_returns_and_kl_semantics_preserved(self):
        aux, _ = _aux_with_logp()
        aux = apply_aux_sft_supervision(aux, beta=0.1, use_kl_loss=True,
                                        noise_sign=True, noise_seed=2)
        assert torch.equal(aux.batch['returns'], aux.batch['advantages'])
        assert torch.equal(aux.batch['ref_log_prob'], aux.batch['old_log_probs'])


class TestHalfGold:
    def test_wrapper_tags_untouched_mask_span_valid(self):
        aux, tok = _aux_with_logp(placebo_mode='half_gold', seed=3)
        text = tok.decode(aux.batch['responses'][0])
        assert '<predict>' in text and '</predict>' in text  # 标签不动
        assert text.startswith("<think>hmm</think>")          # 前缀保留
        assert '<action>go</action>' in text                  # 后缀保留
        assert aux.batch['aux_token_mask'].sum() > 0

    def test_corruption_rate_close_to_p(self):
        rng = np.random.RandomState(0)
        ids = list(range(100, 1100))  # 1000 tokens
        out = _corrupt_token_ids(ids, rng, 0.5, vocab_size=1000,
                                 special_ids=frozenset([0, 3]))
        assert len(out) == len(ids)
        changed = sum(1 for a, b in zip(ids, out) if a != b)
        assert 400 < changed < 600  # p=0.5 ± 统计噪声

    def test_p_zero_is_identity_p_one_full(self):
        rng = np.random.RandomState(1)
        ids = [ord(c) for c in GOLD]
        assert _corrupt_token_ids(ids, rng, 0.0, 1000, frozenset()) == ids
        out = _corrupt_token_ids(list(range(500, 540)), rng, 1.0, 1000, frozenset())
        # p=1 时几乎全换 (随机撞回原值概率 1/1000 量级)
        changed = sum(1 for a, b in zip(range(500, 540), out) if a != b)
        assert changed >= 38

    def test_no_special_tokens_injected(self):
        rng = np.random.RandomState(2)
        out = _corrupt_token_ids(list(range(100, 400)), rng, 1.0, vocab_size=1000,
                                 special_ids=frozenset([0, 3]))
        assert 0 not in out and 3 not in out

    def test_step_seed_gives_fresh_corruption(self):
        texts = []
        for seed in (10, 10, 11):
            aux, tok = _aux_with_logp(placebo_mode='half_gold', seed=seed)
            texts.append(tok.decode(aux.batch['responses'][0]))
        assert texts[0] == texts[1]      # 同步内可复现
        assert texts[0] != texts[2]      # 换步新鲜采样

    def test_gold_mode_unaffected_by_new_param(self):
        aux, tok = _aux_with_logp(placebo_mode=None, half_gold_p=0.9)
        text = tok.decode(aux.batch['responses'][0])
        assert f"<predict>{GOLD}</predict>" in text  # 非 half_gold 模式不腐蚀
