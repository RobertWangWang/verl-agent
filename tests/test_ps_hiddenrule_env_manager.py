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
HRG 版 PS 特征提取器与 manager 预测采集测试 (真环境,单实例假并行,无 ray/GPU)

运行: pytest tests/test_ps_hiddenrule_env_manager.py -v
"""

import re

import pytest
from omegaconf import OmegaConf

from agent_system.environments.env_manager import HiddenRuleEnvironmentManager
from agent_system.environments.env_package.hiddenrule.features import (
    HRG_OBJECT_VOCAB,
    HRGDeviceStateFeature,
    HRGLocationFeature,
    HRGVisibleObjectsF1Feature,
    _visible_names,
    create_hiddenrule_schema_extractor,
)
from agent_system.environments.env_package.hiddenrule.hiddenrule import HiddenRuleEnv, HRGConfig
from agent_system.environments.verifiable_features import (
    VerifiableFeature,
    parse_predict_block,
    prediction_to_features,
)

SAMPLE_OBS = ("Last action result: You enter the rooms.\n"
              "You are in room 2 of 5. The vault is in room 4.\n"
              "Devices here: lever_A is up; dial_B is set to 2.\n"
              "Doors from here lead to: room 1 (open), room 3 (open).\n"
              "You see: note_1, brass_key.\n"
              "Inventory: [].")


# ---------------------------------------------------------------------------
# 提取器单测
# ---------------------------------------------------------------------------

class TestHRGExtractors:
    def test_location(self):
        feat = HRGLocationFeature().extract(SAMPLE_OBS, [], {})
        assert feat.value == 'room 2'

    def test_visible_names(self):
        names = _visible_names(SAMPLE_OBS)
        assert names == {'lever_a', 'dial_b', 'note_1', 'brass_key'}

    def test_device_state_pairs(self):
        feat = HRGDeviceStateFeature().extract(SAMPLE_OBS, [], {})
        assert ('lever_a', 'up') in feat.value['pairs']
        assert ('dial_b', 'set to 2') in feat.value['pairs']

    def test_f1_partial(self):
        ext = HRGVisibleObjectsF1Feature()
        actual = ext.extract(SAMPLE_OBS, [], {})
        pred = VerifiableFeature('visible_objects', {'objects': ['lever_a', 'note_1']})
        # tp=2, precision=1.0, recall=2/4 → F1 = 2/3
        assert ext.verify_score(pred, actual) == pytest.approx(2 / 3)

    def test_schema_extractor_full_accuracy(self):
        composite = create_hiddenrule_schema_extractor()
        actual = composite.extract_all(SAMPLE_OBS, [], {'won': False})
        predicted = prediction_to_features(
            {'next_location': 'room 2', 'objects_visible': True, 'task_done': False})
        assert composite.compute_reward(predicted, actual) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# parse_predict_block 的 HRG 词表
# ---------------------------------------------------------------------------

def test_parse_visible_objects_with_hrg_vocab():
    text = "<predict>next_location: room 3; objects_visible: yes; visible_objects: lever_A, note_1, dragon; task_done: no</predict>"
    parsed = parse_predict_block(text, object_vocab=HRG_OBJECT_VOCAB)
    assert parsed['visible_objects'] == ['lever_a', 'note_1']  # dragon 不在词表被滤掉
    assert parsed['next_location'] == 'room 3'


# ---------------------------------------------------------------------------
# manager 预测采集流程 (真环境)
# ---------------------------------------------------------------------------

class SingleEnvWrapper:
    def __init__(self, cfg, seed=0):
        self.env = HiddenRuleEnv(cfg)
        self._seed = seed

    def reset(self):
        obs, info = self.env.reset(self._seed)
        return [obs], [info]

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions[0])
        return [obs], [reward], [done], [info]

    @property
    def get_admissible_commands(self):
        return [self.env.admissible_actions()]


def projection(text_actions, admissible):
    actions = []
    for t in text_actions:
        m = re.search(r'<action>(.*?)</action>', t, re.DOTALL)
        actions.append(m.group(1).strip() if m else t)
    return actions, [1] * len(actions)


def make_manager(prediction_enable=True, seed=0):
    config = OmegaConf.create({
        'env': {
            'env_name': 'hiddenrule/HiddenRuleEnv',
            'history_length': 2,
            'hiddenrule': {
                'prediction': {'enable': prediction_enable, 'horizon': 1,
                               'feature_weights': None},
            },
        },
    })
    wrapper = SingleEnvWrapper(HRGConfig(key_door_prob=0.0), seed=seed)
    return HiddenRuleEnvironmentManager(wrapper, projection, config)


def response(predict_block, action):
    return f"<think>...</think>{predict_block}<action>{action}</action>"


class TestManagerPrediction:
    def test_prompt_has_predict_block(self):
        manager = make_manager()
        obs, _ = manager.reset(kwargs=None)
        assert '<predict>' in obs['text'][0]
        assert 'objects_visible' in obs['text'][0]

    def test_correct_prediction_after_look(self):
        """look 不改变世界: 用 reset 后的原始观测构造必然正确的预测"""
        manager = make_manager()
        manager.reset(kwargs=None)
        raw = manager.pre_text_obs[0]
        room = HRGLocationFeature().extract(raw, [], {}).value
        visible = 'yes' if _visible_names(raw) else 'no'
        _, _, _, infos = manager.step([response(
            f"<predict>next_location: {room}; objects_visible: {visible}; task_done: no</predict>",
            "look")])
        assert infos[0]['pred_parse_valid'] is True
        assert infos[0]['pred_accuracy'] == pytest.approx(1.0)
        assert infos[0]['pred_reward'] == pytest.approx(1.0)  # Φ_1 − Φ_0

    def test_wrong_prediction(self):
        manager = make_manager()
        manager.reset(kwargs=None)
        raw = manager.pre_text_obs[0]
        room = HRGLocationFeature().extract(raw, [], {}).value
        wrong_room = 'room 99'
        wrong_visible = 'no' if _visible_names(raw) else 'yes'
        _, _, _, infos = manager.step([response(
            f"<predict>next_location: {wrong_room}; objects_visible: {wrong_visible}; task_done: no</predict>",
            "look")])
        assert infos[0]['pred_accuracy'] == pytest.approx(0.0)

    def test_f1_logged(self):
        manager = make_manager()
        manager.reset(kwargs=None)
        raw = manager.pre_text_obs[0]
        names = sorted(_visible_names(raw))
        pred_names = ', '.join(names) if names else 'none'
        _, _, _, infos = manager.step([response(
            f"<predict>next_location: none; objects_visible: yes; "
            f"visible_objects: {pred_names}; task_done: no</predict>",
            "look")])
        assert infos[0]['pred_f1_visible_objects'] == pytest.approx(1.0 if names else 1.0)

    def test_prediction_disabled_clean(self):
        manager = make_manager(prediction_enable=False)
        obs, _ = manager.reset(kwargs=None)
        assert '<predict>' not in obs['text'][0]
        _, _, _, infos = manager.step([response("", "look")])
        assert 'pred_reward' not in infos[0]
