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

"""S6b: 辅助 SFT 批构造与监督写入 (docs/ps_grpo_integration_design.md §8.2)"""

import numpy as np
import pytest
import torch

from agent_system.multi_turn_rollout.aux_sft import (
    apply_aux_sft_supervision,
    build_aux_sft_batch,
)
from verl import DataProto


class CharTokenizer:
    """字符级 mock tokenizer: id = ord(char); 可逆,便于精确断言"""

    pad_token_id = 0
    eos_token_id = 3

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


class TestBuildAuxBatch:
    def test_basic_replacement_and_span(self):
        batch, tok = _make_batch([RESP], [GOLD])
        aux = build_aux_sft_batch(batch, tok)
        assert aux is not None and len(aux) == 1
        text = tok.decode(aux.batch['responses'][0])
        assert f"<predict>{GOLD}</predict>" in text
        assert "room 9" not in text                       # 原预测被替换
        assert text.startswith("<think>hmm</think>")       # 前缀保留
        assert text.rstrip('\x00').endswith("<action>go</action>")  # 后缀保留
        # span 精确覆盖 gold 块 (含标签)
        mask = aux.batch['aux_token_mask'][0]
        span_ids = aux.batch['responses'][0][mask.bool()]
        assert tok.decode(span_ids) == f"<predict>{GOLD}</predict>"

    def test_rows_without_predict_block_skipped(self):
        batch, tok = _make_batch(["<action>go</action>", RESP], [GOLD, GOLD])
        aux = build_aux_sft_batch(batch, tok)
        assert len(aux) == 1

    def test_rows_without_gold_skipped(self):
        batch, tok = _make_batch([RESP, RESP], ['', GOLD])
        aux = build_aux_sft_batch(batch, tok)
        assert len(aux) == 1

    def test_no_candidates_returns_none(self):
        batch, tok = _make_batch(["<action>go</action>"], [''])
        assert build_aux_sft_batch(batch, tok) is None

    def test_no_gold_key_returns_none(self):
        batch, tok = _make_batch([RESP], [GOLD])
        del batch.non_tensor_batch['gold_predict']
        assert build_aux_sft_batch(batch, tok) is None

    def test_stale_parent_tensors_not_inherited(self):
        """父行的 old_log_probs/advantages 与旧 response 对齐,禁止带入"""
        batch, tok = _make_batch([RESP], [GOLD])
        aux = build_aux_sft_batch(batch, tok)
        assert 'old_log_probs' not in aux.batch.keys()
        assert 'advantages' not in aux.batch.keys()

    def test_attention_and_positions_consistent(self):
        batch, tok = _make_batch([RESP], [GOLD])
        aux = build_aux_sft_batch(batch, tok)
        attn = aux.batch['attention_mask'][0]
        pos = aux.batch['position_ids'][0]
        n_resp = int(attn[6:].sum())
        assert n_resp == int((aux.batch['responses'][0] != 0).sum())
        # 有效位置单调 +1
        active = pos[attn.bool()]
        assert torch.all(active[1:] - active[:-1] == 1)

    def test_fraction_subsamples(self):
        batch, tok = _make_batch([RESP] * 8, [GOLD] * 8)
        aux = build_aux_sft_batch(batch, tok, fraction=0.5, seed=1)
        assert len(aux) == 4

    def test_truncation_at_resp_len(self):
        long_gold = "next_location: " + "room 3, " * 40
        batch, tok = _make_batch([RESP], [long_gold], resp_len=64)
        aux = build_aux_sft_batch(batch, tok)
        assert aux.batch['responses'].shape[1] == 64
        assert int(aux.batch['aux_token_mask'][0].sum()) <= 64


class TestApplySupervision:
    def test_advantage_arithmetic_and_kl_neutralization(self):
        batch, tok = _make_batch([RESP], [GOLD])
        aux = build_aux_sft_batch(batch, tok)
        aux.batch['old_log_probs'] = torch.randn(1, aux.batch['responses'].shape[1])
        aux = apply_aux_sft_supervision(aux, beta=0.25, use_kl_loss=True)
        mask = aux.batch['aux_token_mask']
        assert torch.equal(aux.batch['advantages'], 0.25 * mask)
        assert torch.equal(aux.batch['returns'], 0.25 * mask)
        assert torch.equal(aux.batch['ref_log_prob'], aux.batch['old_log_probs'])
        # gold 段外优势为 0
        assert float(aux.batch['advantages'][0][~mask[0].bool()].abs().sum()) == 0.0
