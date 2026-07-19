# Copyright 2025 Nanyang Technological University (NTU), Singapore
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
PS-GRPO S2 测试: trainer 端预测奖励注入与 λ 退火调度

运行: pytest tests/test_ps_reward_injection.py -v
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import apply_prediction_reward, pred_lambda_schedule

PROMPT_LEN = 4
RESPONSE_LEN = 5


def make_batch(pred_rewards, valid_response_lengths, with_step_rewards=False):
    """构造最小的 rollout 批: 3 个 step 样本，valid response 长度各不相同"""
    bs = len(pred_rewards)
    attention_mask = torch.zeros(bs, PROMPT_LEN + RESPONSE_LEN, dtype=torch.long)
    attention_mask[:, :PROMPT_LEN] = 1
    for i, vrl in enumerate(valid_response_lengths):
        attention_mask[i, PROMPT_LEN:PROMPT_LEN + vrl] = 1

    tensors = {
        'prompts': torch.zeros(bs, PROMPT_LEN, dtype=torch.long),
        'responses': torch.zeros(bs, RESPONSE_LEN, dtype=torch.long),
        'attention_mask': attention_mask,
        'token_level_scores': torch.zeros(bs, RESPONSE_LEN, dtype=torch.float32),
    }
    if with_step_rewards:
        tensors['step_rewards'] = torch.zeros(bs, dtype=torch.float32)

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors={'pred_rewards': np.array(pred_rewards, dtype=np.float32)},
    )


# ---------------------------------------------------------------------------
# apply_prediction_reward
# ---------------------------------------------------------------------------

class TestApplyPredictionReward:
    def test_injects_at_last_valid_response_token(self):
        data = make_batch(pred_rewards=[1.0, -0.5, 0.0], valid_response_lengths=[5, 3, 1])
        data, metrics = apply_prediction_reward(data, lambda_pred=0.1)

        scores = data.batch['token_level_scores']
        assert scores[0, 4].item() == pytest.approx(0.1)      # λ * 1.0
        assert scores[1, 2].item() == pytest.approx(-0.05)    # λ * -0.5，负 shaping 也注入
        assert scores[2, 0].item() == pytest.approx(0.0)
        # 除最后有效 token 外全为 0
        assert scores.abs().sum().item() == pytest.approx(0.15)

    def test_adds_on_top_of_existing_scores(self):
        """与 EpisodeRewardManager 已写入的 episode 总分叠加，而非覆盖"""
        data = make_batch(pred_rewards=[1.0], valid_response_lengths=[5])
        data.batch['token_level_scores'][0, 4] = 10.0  # episode 总分
        data, _ = apply_prediction_reward(data, lambda_pred=0.1)
        assert data.batch['token_level_scores'][0, 4].item() == pytest.approx(10.1)

    def test_updates_gigpo_step_rewards_when_present(self):
        data = make_batch(pred_rewards=[1.0, -0.5], valid_response_lengths=[5, 3],
                          with_step_rewards=True)
        data, _ = apply_prediction_reward(data, lambda_pred=0.2)
        assert data.batch['step_rewards'][0].item() == pytest.approx(0.2)
        assert data.batch['step_rewards'][1].item() == pytest.approx(-0.1)

    def test_metrics(self):
        data = make_batch(pred_rewards=[1.0, 0.0], valid_response_lengths=[5, 3])
        _, metrics = apply_prediction_reward(data, lambda_pred=0.1)
        assert metrics['episode/pred_lambda'] == pytest.approx(0.1)
        assert metrics['episode/pred_reward_injected/mean'] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# pred_lambda_schedule
# ---------------------------------------------------------------------------

def make_cfg(**kwargs):
    cfg = {'enable': True, 'lambda': 0.1,
           'anneal': {'style': 'constant', 'final_ratio': 0.0, 'total_steps': -1}}
    cfg['anneal'].update(kwargs)
    return OmegaConf.create(cfg)


class TestPredLambdaSchedule:
    def test_constant(self):
        cfg = make_cfg(style='constant')
        assert pred_lambda_schedule(cfg, 0, 100) == pytest.approx(0.1)
        assert pred_lambda_schedule(cfg, 100, 100) == pytest.approx(0.1)

    def test_linear(self):
        cfg = make_cfg(style='linear', final_ratio=0.0, total_steps=100)
        assert pred_lambda_schedule(cfg, 0, -1) == pytest.approx(0.1)
        assert pred_lambda_schedule(cfg, 50, -1) == pytest.approx(0.05)
        assert pred_lambda_schedule(cfg, 100, -1) == pytest.approx(0.0)
        assert pred_lambda_schedule(cfg, 150, -1) == pytest.approx(0.0)  # 超出后不为负

    def test_linear_with_final_ratio(self):
        cfg = make_cfg(style='linear', final_ratio=0.2, total_steps=100)
        assert pred_lambda_schedule(cfg, 100, -1) == pytest.approx(0.02)

    def test_cosine_boundaries(self):
        cfg = make_cfg(style='cosine', final_ratio=0.0, total_steps=100)
        assert pred_lambda_schedule(cfg, 0, -1) == pytest.approx(0.1)
        assert pred_lambda_schedule(cfg, 50, -1) == pytest.approx(0.05)
        assert pred_lambda_schedule(cfg, 100, -1) == pytest.approx(0.0, abs=1e-9)

    def test_total_steps_falls_back_to_training_steps(self):
        cfg = make_cfg(style='linear', total_steps=-1)
        assert pred_lambda_schedule(cfg, 50, 100) == pytest.approx(0.05)

    def test_no_total_available_returns_base(self):
        """退火需要总步数; 完全拿不到时退化为常数，不抛错"""
        cfg = make_cfg(style='linear', total_steps=-1)
        assert pred_lambda_schedule(cfg, 50, 0) == pytest.approx(0.1)

    def test_unknown_style_raises(self):
        cfg = make_cfg(style='exponential')
        with pytest.raises(ValueError):
            pred_lambda_schedule(cfg, 0, 100)

    def test_missing_anneal_block_is_constant(self):
        cfg = OmegaConf.create({'enable': True, 'lambda': 0.3})
        assert pred_lambda_schedule(cfg, 50, 100) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# R38: GDPO 式分通道解耦优势 (apply_decoupled_pred_advantage)
# ---------------------------------------------------------------------------

class TestDecoupledPredAdvantage:
    def _batch(self, uids, pred_rewards, resp_len=4, prompt_len=2):
        import numpy as np
        import torch
        from tensordict import TensorDict
        from verl import DataProto
        n = len(uids)
        attn = torch.ones(n, prompt_len + resp_len, dtype=torch.long)
        attn[:, -1] = 0  # 最后一个 response 位为 padding, 验证 mask 尊重
        td = TensorDict({
            'responses': torch.zeros(n, resp_len, dtype=torch.long),
            'attention_mask': attn,
            'advantages': torch.zeros(n, resp_len),
        }, batch_size=[n])
        return DataProto(batch=td, non_tensor_batch={
            'uid': np.array(uids, dtype=object),
            'pred_rewards': np.array(pred_rewards, dtype=np.float32),
        })

    def test_group_centering_and_lambda_cap(self):
        import torch
        from verl.trainer.ppo.ray_trainer import apply_decoupled_pred_advantage
        data = self._batch(['a', 'a', 'b', 'b'], [0.1, 0.0, 0.5, 0.5])
        data, m = apply_decoupled_pred_advantage(data, lambda_pred=0.1, norm_by_std=True)
        adv = data.batch['advantages']
        # 组 a: 差异被组内归一化到 ±1, 再乘 λ → ±0.1 (封顶在 λ·O(1))
        assert adv[0, 0] == pytest.approx(0.1, abs=1e-4)
        assert adv[1, 0] == pytest.approx(-0.1, abs=1e-4)
        # 组 b: 组内无差异 → 贡献 0 (跨组的 0.5 vs 0.05 不泄漏)
        assert float(adv[2].abs().sum()) == pytest.approx(0.0, abs=1e-6)
        # padding 位不携带优势
        assert float(adv[0, -1]) == 0.0
        assert m['episode/pred_adv_contrib/max_abs'] <= 0.1 + 1e-4

    def test_mean_only_mode(self):
        from verl.trainer.ppo.ray_trainer import apply_decoupled_pred_advantage
        data = self._batch(['a', 'a'], [0.3, 0.1])
        data, _ = apply_decoupled_pred_advantage(data, lambda_pred=1.0, norm_by_std=False)
        adv = data.batch['advantages']
        assert adv[0, 0] == pytest.approx(0.1)   # 0.3 - 0.2
        assert adv[1, 0] == pytest.approx(-0.1)

    def test_additive_on_existing_task_advantage(self):
        import torch
        from verl.trainer.ppo.ray_trainer import apply_decoupled_pred_advantage
        data = self._batch(['a', 'a'], [1.0, 0.0])
        data.batch['advantages'][:] = 2.0  # 既有任务优势
        data, _ = apply_decoupled_pred_advantage(data, lambda_pred=0.1, norm_by_std=True)
        adv = data.batch['advantages']
        assert adv[0, 0] == pytest.approx(2.1, abs=1e-4)
        assert adv[1, 0] == pytest.approx(1.9, abs=1e-4)
        # padding 位维持原任务优势不被 pred 贡献污染
        assert adv[0, -1] == pytest.approx(2.0)

    def test_amplification_capped_vs_legacy(self):
        """核心主张: 全败组内微小 pred 差异的优势贡献 ≤ λ, 而非满幅"""
        from verl.trainer.ppo.ray_trainer import apply_decoupled_pred_advantage
        data = self._batch(['g'] * 4, [0.002, 0.001, 0.0015, 0.0005])
        data, m = apply_decoupled_pred_advantage(data, lambda_pred=0.1, norm_by_std=True)
        assert m['episode/pred_adv_contrib/max_abs'] <= 0.1 * 1.5 + 1e-4  # std 归一后 |z|≲1.5
