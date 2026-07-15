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
PS-GRPO S1 接入测试: AlfWorldEnvironmentManager 的预测采集流程 (fake envs，无需安装 ALFWorld)

运行: pytest tests/test_ps_alfworld_env_manager.py -v
"""

import re

import pytest
from omegaconf import OmegaConf

from agent_system.environments.env_manager import AlfWorldEnvironmentManager
from agent_system.memory import HybridMemory, SimpleMemory

TASK_OBS = ("-= Welcome to TextWorld, ALFRED! =-\n\n"
            "You are in the middle of a room. Your task is to: put a clean ladle in diningtable.")


class FakeAlfWorldEnvs:
    """两步脚本化环境: cabinet 1 (关着，看不见目标) → 打开后看见 ladle"""

    def __init__(self):
        self._step_count = 0
        self.get_admissible_commands = [['go to cabinet 1', 'look']]

    def reset(self):
        self._step_count = 0
        self.get_admissible_commands = [['go to cabinet 1', 'look']]
        return [TASK_OBS], None, [{'won': False, 'extra.gamefile': 'json_2.1.1/train/pick_clean_then_place_in_recep-Ladle/trial_1'}]

    def step(self, actions):
        self._step_count += 1
        if self._step_count == 1:
            text_obs = ["You arrive at cabinet 1. The cabinet 1 is closed."]
            self.get_admissible_commands = [['open cabinet 1', 'examine cabinet 1', 'go to diningtable 1']]
        else:
            text_obs = ["You open the cabinet 1. The cabinet 1 is open. In it, you see a ladle."]
            self.get_admissible_commands = [['close cabinet 1', 'pick up ladle 1']]
        infos = [{'won': False, 'extra.gamefile': None}]
        return text_obs, None, [0.0], [False], infos


def fake_projection(text_actions, admissible_commands):
    actions, valids = [], []
    for text in text_actions:
        match = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
        actions.append(match.group(1).strip() if match else text)
        valids.append(1 if match else 0)
    return actions, valids


def make_config(prediction_enable, feature_protocol='task_targets', feature_weights='legacy'):
    if feature_weights == 'legacy':
        feature_weights = {
            'object_seen': 0.4,
            'location_change': 0.4,
            'action_available': 0.2,
            'task_progress': 0.0,
        }
    return OmegaConf.create({
        'env': {
            'env_name': 'alfworld/AlfredTWEnv',
            'history_length': 2,
            'alfworld': {
                'eval_dataset': 'eval_in_distribution',
                'prediction': {
                    'enable': prediction_enable,
                    'horizon': 1,
                    'feature_protocol': feature_protocol,
                    'feature_weights': feature_weights,
                },
            },
        },
    })


def make_manager(prediction_enable=True, feature_protocol='task_targets', feature_weights='legacy'):
    return AlfWorldEnvironmentManager(
        FakeAlfWorldEnvs(), fake_projection,
        make_config(prediction_enable, feature_protocol, feature_weights))


def response(predict_block, action):
    think = "<think>reasoning...</think>"
    return f"{think}{predict_block}<action>{action}</action>"


# ---------------------------------------------------------------------------
# 预测开启时的完整流程
# ---------------------------------------------------------------------------

class TestPredictionEnabled:
    def test_reset_builds_ps_prompt_and_task_extractors(self):
        manager = make_manager()
        obs, _ = manager.reset(kwargs=None)

        assert isinstance(manager.memory, HybridMemory)
        assert '<predict>' in obs['text'][0]
        # 目标物体集来自任务描述，与 prompt 的 target_visible 语义对齐
        targets = manager.feature_extractors[0].extractors[0][0].target_objects
        assert targets == {'ladle', 'diningtable'}

    def test_correct_prediction_full_accuracy(self):
        manager = make_manager()
        manager.reset(kwargs=None)

        # 预测: 去 cabinet 1，目标物体不可见 (柜子是关的)，任务未完成 → 全对
        _, _, _, infos = manager.step([response(
            "<predict>next_location: cabinet 1; target_visible: no; task_done: no</predict>",
            "go to cabinet 1")])

        assert infos[0]['pred_parse_valid'] is True
        assert infos[0]['pred_accuracy'] == pytest.approx(1.0)
        # potential shaping: Φ_1 − Φ_0 = 1.0 − 0.0 (环境侧未加权，λ 由 trainer 施加)
        assert infos[0]['pred_reward'] == pytest.approx(1.0)

    def test_wrong_prediction_zero_accuracy(self):
        manager = make_manager()
        manager.reset(kwargs=None)

        # 预测: 位置错 (实际 cabinet 1)、可见性错 (实际不可见)
        _, _, _, infos = manager.step([response(
            "<predict>next_location: diningtable 1; target_visible: yes; task_done: no</predict>",
            "go to cabinet 1")])

        assert infos[0]['pred_parse_valid'] is True
        assert infos[0]['pred_accuracy'] == pytest.approx(0.0)

    def test_missing_predict_block_counts_as_parse_failure(self):
        manager = make_manager()
        manager.reset(kwargs=None)

        # 第一步预测全对 (Φ_1 = 1.0)
        manager.step([response(
            "<predict>next_location: cabinet 1; target_visible: no; task_done: no</predict>",
            "go to cabinet 1")])
        # 第二步缺 <predict> 块 → Φ_2 = 0，shaping 给出负奖励 Φ_2 − Φ_1 = −1
        _, _, _, infos = manager.step([response("", "open cabinet 1")])

        assert infos[0]['pred_parse_valid'] is False
        assert infos[0]['pred_accuracy'] == pytest.approx(0.0)
        assert infos[0]['pred_reward'] == pytest.approx(-1.0)

    def test_second_step_sees_target_and_ps_history_template(self):
        manager = make_manager()
        manager.reset(kwargs=None)
        manager.step([response(
            "<predict>next_location: cabinet 1; target_visible: no; task_done: no</predict>",
            "go to cabinet 1")])

        # 打开柜子后看见 ladle → target_visible: yes 应验证为对; 位置无变化 → none
        next_obs, _, _, infos = manager.step([response(
            "<predict>next_location: none; target_visible: yes; task_done: no</predict>",
            "open cabinet 1")])

        assert infos[0]['pred_accuracy'] == pytest.approx(1.0)
        # 带历史的 PS 模板: 含任务描述与 <predict> 指令
        assert '<predict>' in next_obs['text'][0]
        assert 'put a clean ladle in diningtable' in next_obs['text'][0]


# ---------------------------------------------------------------------------
# 预测关闭时行为与 baseline 完全一致
# ---------------------------------------------------------------------------

class TestPredictionDisabled:
    def test_baseline_untouched(self):
        manager = make_manager(prediction_enable=False)
        obs, _ = manager.reset(kwargs=None)

        assert isinstance(manager.memory, SimpleMemory)
        assert '<predict>' not in obs['text'][0]

        next_obs, rewards, dones, infos = manager.step([response("", "go to cabinet 1")])
        assert 'pred_reward' not in infos[0]
        assert 'pred_accuracy' not in infos[0]
        assert '<predict>' not in next_obs['text'][0]

    def test_missing_prediction_config_key_defaults_to_disabled(self):
        """旧配置里没有 env.alfworld.prediction 键时不应报错"""
        config = OmegaConf.create({
            'env': {
                'env_name': 'alfworld/AlfredTWEnv',
                'history_length': 2,
                'alfworld': {'eval_dataset': 'eval_in_distribution'},
            },
        })
        manager = AlfWorldEnvironmentManager(FakeAlfWorldEnvs(), fake_projection, config)
        assert manager.pred_enabled is False
        assert isinstance(manager.memory, SimpleMemory)


# ---------------------------------------------------------------------------
# alfworld_projection 的 require_think 开关 (Qwen3 enable_thinking=False 兼容)
# ---------------------------------------------------------------------------

class TestProjectionRequireThink:
    QWEN3_STYLE = "### Reasoning:\nI should check the drawer.\n<action>examine drawer 2</action>"
    QWEN25_STYLE = "<think>I should check the drawer.</think>\n<action>examine drawer 2</action>"

    def _project(self, text, require_think):
        from agent_system.environments.env_package.alfworld.projection import alfworld_projection
        actions, valids = alfworld_projection([text], [["examine drawer 2"]],
                                              require_think=require_think)
        return actions[0], valids[0]

    def test_legacy_mode_rejects_missing_think(self):
        action, valid = self._project(self.QWEN3_STYLE, require_think=True)
        assert action == "examine drawer 2"  # 动作仍被提取
        assert valid == 0                    # 但判无效 (旧语义)

    def test_relaxed_mode_accepts_qwen3_format(self):
        action, valid = self._project(self.QWEN3_STYLE, require_think=False)
        assert action == "examine drawer 2"
        assert valid == 1

    def test_both_modes_accept_think_format(self):
        for rt in (True, False):
            action, valid = self._project(self.QWEN25_STYLE, require_think=rt)
            assert action == "examine drawer 2"
            assert valid == 1

    def test_relaxed_mode_still_rejects_missing_action(self):
        _, valid = self._project("### Reasoning: no action tag here", require_think=False)
        assert valid == 0


# ---------------------------------------------------------------------------
# v0.2 schema 协议模式 (S4a, docs/ps_grpo_integration_design.md §7)
# ---------------------------------------------------------------------------

class TestSchemaProtocol:
    def make(self):
        return make_manager(prediction_enable=True, feature_protocol='schema',
                            feature_weights=None)

    def test_shared_task_agnostic_extractor(self):
        """schema 模式: 所有环境共享同一个 Φ 实例,无 per-task 目标集"""
        manager = self.make()
        manager.reset(kwargs=None)
        assert manager.feature_extractors[0] is manager.feature_extractors[-1]
        first_extractor = manager.feature_extractors[0].extractors[0][0]
        assert not hasattr(first_extractor, 'target_objects')

    def test_prompt_uses_schema_predict_fields(self):
        manager = self.make()
        obs, _ = manager.reset(kwargs=None)
        assert 'objects_visible' in obs['text'][0]
        assert 'visible_objects' in obs['text'][0]
        assert 'target_visible' not in obs['text'][0]

    def test_correct_schema_prediction(self):
        manager = self.make()
        manager.reset(kwargs=None)
        # 去 cabinet 1 (关着) → 看不到任何物体
        _, _, _, infos = manager.step([response(
            "<predict>next_location: cabinet 1; objects_visible: no; "
            "visible_objects: none; task_done: no</predict>",
            "go to cabinet 1")])
        assert infos[0]['pred_parse_valid'] is True
        assert infos[0]['pred_accuracy'] == pytest.approx(1.0)

    def test_wrong_objects_visible_partial_score(self):
        manager = self.make()
        manager.reset(kwargs=None)
        # 位置对 (0.5✓), objects_visible 错 (0.5✗) → 0.5
        _, _, _, infos = manager.step([response(
            "<predict>next_location: cabinet 1; objects_visible: yes; task_done: no</predict>",
            "go to cabinet 1")])
        assert infos[0]['pred_accuracy'] == pytest.approx(0.5)

    def test_visible_objects_f1_is_logged_not_rewarded(self):
        """开放集 F1 权重为 0: 预测错物体列表不影响加权准确率"""
        manager = self.make()
        manager.reset(kwargs=None)
        manager.step([response(
            "<predict>next_location: cabinet 1; objects_visible: no; task_done: no</predict>",
            "go to cabinet 1")])
        # 第二步开柜见 ladle; visible_objects 全错也不扣加权分
        _, _, _, infos = manager.step([response(
            "<predict>next_location: none; objects_visible: yes; "
            "visible_objects: fridge, knife; task_done: no</predict>",
            "open cabinet 1")])
        assert infos[0]['pred_accuracy'] == pytest.approx(1.0)

    def test_f1_logged_in_info(self):
        manager = self.make()
        manager.reset(kwargs=None)
        manager.step([response(
            "<predict>next_location: cabinet 1; objects_visible: no; "
            "visible_objects: none; task_done: no</predict>",
            "go to cabinet 1")])
        # 第二步实际看见 ladle; 预测 [ladle] → F1 = 1.0
        _, _, _, infos = manager.step([response(
            "<predict>next_location: none; objects_visible: yes; "
            "visible_objects: ladle; task_done: no</predict>",
            "open cabinet 1")])
        assert infos[0]['pred_f1_visible_objects'] == pytest.approx(1.0)

    def test_f1_partial_credit(self):
        manager = self.make()
        manager.reset(kwargs=None)
        manager.step([response(
            "<predict>next_location: cabinet 1; objects_visible: no; task_done: no</predict>",
            "go to cabinet 1")])
        # 实际 {ladle}; 预测 {ladle, knife} → precision 0.5, recall 1.0 → F1 = 2/3
        _, _, _, infos = manager.step([response(
            "<predict>next_location: none; objects_visible: yes; "
            "visible_objects: ladle, knife; task_done: no</predict>",
            "open cabinet 1")])
        assert infos[0]['pred_f1_visible_objects'] == pytest.approx(2 / 3)

    def test_f1_absent_field_scores_zero(self):
        manager = self.make()
        manager.reset(kwargs=None)
        _, _, _, infos = manager.step([response(
            "<predict>next_location: cabinet 1; objects_visible: no; task_done: no</predict>",
            "go to cabinet 1")])
        assert infos[0]['pred_f1_visible_objects'] == pytest.approx(0.0)
