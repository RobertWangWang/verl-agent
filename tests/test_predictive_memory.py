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
预测充分性记忆模块测试

运行: pytest tests/test_predictive_memory.py -v
"""

import pytest

from agent_system.environments.verifiable_features import (
    create_alfworld_feature_extractor,
    parse_predict_block,
    prediction_to_features,
)
from agent_system.memory.predictive_memory import (
    HybridMemory,
    PredictionResult,
    PredictiveMemory,
)


@pytest.fixture
def feature_extractor():
    return create_alfworld_feature_extractor(
        object_types={'ladle', 'knife', 'fridge'},
        feature_weights={
            'object_seen': 0.4,
            'location_change': 0.4,
            'action_available': 0.2,
            'task_progress': 0.0,
        }
    )


# ---------------------------------------------------------------------------
# 存储 / 获取 (SimpleMemory 兼容行为)
# ---------------------------------------------------------------------------

class TestStorage:
    def test_reset_and_store(self):
        memory = PredictiveMemory()
        memory.reset(batch_size=2)
        assert memory.batch_size == 2

        memory.store({
            'text_obs': ['You see a ladle.', 'You see a knife.'],
            'action': ['pick up ladle', 'pick up knife'],
        })
        contexts, valid_lengths = memory.fetch(
            history_length=2, obs_key='text_obs', action_key='action')

        assert len(contexts) == 2
        assert valid_lengths == [1, 1]
        assert 'pick up ladle' in contexts[0]
        assert 'pick up knife' in contexts[1]

    def test_reset_clears_predictions(self):
        memory = PredictiveMemory()
        memory.reset(batch_size=1)
        memory.record_prediction(env_idx=0, turn=1, predicted_features={})
        assert 1 in memory.predictions[0]

        memory.reset(batch_size=1)
        assert memory.predictions[0] == {}
        assert memory.prev_prediction_accuracy == [0.0]


# ---------------------------------------------------------------------------
# record_prediction / verify_prediction (预测来自解析的 <predict> 块)
# ---------------------------------------------------------------------------

class TestPredictionVerification:
    def test_record_and_verify_correct_prediction(self, feature_extractor):
        memory = PredictiveMemory()
        memory.reset(batch_size=1)

        # LLM 在 turn 1 的 response 里给出预测
        response = ("<think>...</think>"
                    "<predict>next_location: cabinet 1; target_visible: yes</predict>"
                    "<action>go to cabinet 1</action>")
        predicted = prediction_to_features(parse_predict_block(response))
        result = memory.record_prediction(env_idx=0, turn=1, predicted_features=predicted)
        assert isinstance(result, PredictionResult)

        # 环境返回 o_{t+1}: 预测完全正确
        accuracy, components = memory.verify_prediction(
            env_idx=0, turn=1,
            feature_extractor=feature_extractor,
            actual_obs="You arrive at cabinet 1. In it, you see a ladle.",
            actual_actions=['pick up ladle'],
            actual_info={'won': False},
        )
        assert accuracy == pytest.approx(1.0)
        assert components == {'location_change': True, 'object_seen': True}

    def test_verify_wrong_prediction(self, feature_extractor):
        memory = PredictiveMemory()
        memory.reset(batch_size=1)

        predicted = prediction_to_features(
            {'next_location': 'fridge 1', 'target_visible': True})
        memory.record_prediction(env_idx=0, turn=1, predicted_features=predicted)

        # 实际到了 cabinet 1 (位置错)，且确实看到 ladle (可见性对) → 加权 0.5
        accuracy, components = memory.verify_prediction(
            env_idx=0, turn=1,
            feature_extractor=feature_extractor,
            actual_obs="You arrive at cabinet 1. In it, you see a ladle.",
            actual_actions=['pick up ladle'],
            actual_info={'won': False},
        )
        assert accuracy == pytest.approx(0.5)
        assert components['location_change'] is False
        assert components['object_seen'] is True

    def test_verify_without_prediction_returns_zero(self, feature_extractor):
        memory = PredictiveMemory()
        memory.reset(batch_size=1)
        accuracy, components = memory.verify_prediction(
            env_idx=0, turn=99,
            feature_extractor=feature_extractor,
            actual_obs="Nothing happens.",
            actual_actions=[],
            actual_info={},
        )
        assert accuracy == 0.0
        assert components == {}


# ---------------------------------------------------------------------------
# compute_prediction_reward (potential-based shaping: Φ_t − Φ_{t−1})
# ---------------------------------------------------------------------------

class TestPredictionReward:
    def test_potential_shaping(self):
        memory = PredictiveMemory(lambda_pred=0.5, use_potential_shaping=True)
        memory.reset(batch_size=1)

        # Turn 1: Φ_1 = 0.8, Φ_0 = 0 → r = 0.5 * (0.8 − 0)
        r1 = memory.compute_prediction_reward(env_idx=0, turn=1, current_accuracy=0.8)
        assert r1.shaped_reward == pytest.approx(0.5 * 0.8)
        assert r1.prediction_accuracy == 0.8

        # Turn 2: Φ_2 = 0.3 → r = 0.5 * (0.3 − 0.8)，预测变差应为负
        r2 = memory.compute_prediction_reward(env_idx=0, turn=2, current_accuracy=0.3)
        assert r2.shaped_reward == pytest.approx(0.5 * (0.3 - 0.8))

    def test_shaping_telescopes_to_final_potential(self):
        """Σ r_pred = λ·(Φ_T − Φ_0): 逐 episode 求和会望远镜相消，
        这正是 r_pred 必须逐 step 注入而不能只加进 episode 总分的原因。"""
        memory = PredictiveMemory(lambda_pred=1.0, use_potential_shaping=True)
        memory.reset(batch_size=1)

        accuracies = [0.2, 0.7, 0.4, 0.9]
        total = sum(
            memory.compute_prediction_reward(0, t, acc).shaped_reward
            for t, acc in enumerate(accuracies)
        )
        assert total == pytest.approx(accuracies[-1] - 0.0)

    def test_without_shaping_uses_raw_accuracy(self):
        memory = PredictiveMemory(lambda_pred=1.0, use_potential_shaping=False)
        memory.reset(batch_size=1)

        memory.compute_prediction_reward(env_idx=0, turn=1, current_accuracy=0.8)
        r2 = memory.compute_prediction_reward(env_idx=0, turn=2, current_accuracy=0.3)
        assert r2.shaped_reward == pytest.approx(0.3)  # 不做差分

    def test_per_env_independent_potentials(self):
        memory = PredictiveMemory(lambda_pred=1.0)
        memory.reset(batch_size=2)

        memory.compute_prediction_reward(env_idx=0, turn=1, current_accuracy=0.9)
        # env 1 的 Φ_0 仍是 0，不受 env 0 影响
        r = memory.compute_prediction_reward(env_idx=1, turn=1, current_accuracy=0.4)
        assert r.shaped_reward == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# HybridMemory
# ---------------------------------------------------------------------------

class TestHybridMemory:
    def test_storage_delegation(self):
        memory = HybridMemory(history_length=2)
        memory.reset(batch_size=2)
        memory.store({'text_obs': ['Obs 1', 'Obs 2'], 'action': ['Action 1', 'Action 2']})

        contexts, lengths = memory.fetch()
        assert len(contexts) == 2
        assert lengths == [1, 1]

    def test_prediction_delegation(self, feature_extractor):
        memory = HybridMemory(history_length=2, enable_prediction=True)
        memory.reset(batch_size=1)

        predicted = prediction_to_features({'next_location': 'cabinet 1'})
        assert memory.record_prediction(0, 1, predicted) is not None

        accuracy, _ = memory.verify_prediction(
            env_idx=0, turn=1,
            feature_extractor=feature_extractor,
            actual_obs="You arrive at cabinet 1.",
            actual_actions=[],
            actual_info={'won': False},
        )
        assert accuracy == pytest.approx(1.0)

        reward = memory.compute_prediction_reward(env_idx=0, turn=1, current_accuracy=accuracy)
        assert reward is not None

    def test_prediction_disabled(self, feature_extractor):
        memory = HybridMemory(history_length=2, enable_prediction=False)
        memory.reset(batch_size=1)

        assert memory.record_prediction(0, 1, {}) is None
        assert memory.verify_prediction(
            env_idx=0, turn=1, feature_extractor=feature_extractor,
            actual_obs="x", actual_actions=[], actual_info={}) == (0.0, {})
        assert memory.compute_prediction_reward(0, 1, 0.5) is None
        assert memory.get_prediction_rewards() == []


# ---------------------------------------------------------------------------
# 记忆摘要
# ---------------------------------------------------------------------------

def test_summary_generation():
    memory = PredictiveMemory()
    memory.reset(batch_size=1)

    for obs, action in [
        ('You arrive at cabinet 1.', 'open cabinet 1'),
        ('You see a ladle.', 'pick up ladle'),
        ('You pick up the ladle.', 'go to diningtable'),
    ]:
        memory.store({'text_obs': [obs], 'action': [action]})

    summary = memory.get_summary_for_prompt(env_idx=0, history_length=3)
    assert 'open cabinet 1' in summary


def test_summary_empty_history():
    memory = PredictiveMemory()
    memory.reset(batch_size=1)
    assert memory.get_summary_for_prompt(env_idx=0) == "No history yet."
