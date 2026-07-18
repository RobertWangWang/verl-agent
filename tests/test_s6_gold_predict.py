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

"""S6a: gold 预测串构造 (docs/ps_grpo_integration_design.md §8.1)

核心不变量: parse_predict_block(gold) 后对同一 actual_features 计
compute_reward 恒为 1.0 —— 监督目标与 RL 奖励目标严格同源。
"""

import pytest

from agent_system.environments.env_package.hiddenrule.features import (
    HRG_OBJECT_VOCAB,
    create_hiddenrule_schema_extractor,
)
from agent_system.environments.verifiable_features import (
    create_alfworld_schema_extractor,
    gold_predict_string,
    parse_predict_block,
    prediction_to_features,
)

ALFWORLD_OBS = (
    "You arrive at cabinet 2. The cabinet 2 is open. In it, you see a mug 1, "
    "and a plate 2."
)
HRG_OBS = (
    "You are in room 3. Devices here: [lever_a: up, dial_b: 2]. "
    "lever_a is up. dial_b is set to 2. "
    "You see: [note_1, brass_key]. Inventory: []."
)


class TestGoldString:
    def test_alfworld_format(self):
        extractor = create_alfworld_schema_extractor()
        actual = extractor.extract_all(ALFWORLD_OBS, [], {'won': False})
        gold = gold_predict_string(actual)
        assert gold.startswith('next_location: ')
        assert 'objects_visible: yes' in gold
        assert 'task_done: no' in gold

    def test_hrg_format_with_devices(self):
        extractor = create_hiddenrule_schema_extractor()
        actual = extractor.extract_all(HRG_OBS, [], {'won': False})
        gold = gold_predict_string(actual, include_device_states=True)
        assert 'next_location: room 3' in gold
        assert 'device_states: ' in gold
        assert 'dial_b=2' in gold          # 'set to 2' 还原为指令词形
        assert 'lever_a=up' in gold

    def test_empty_scene_renders_none(self):
        extractor = create_alfworld_schema_extractor()
        actual = extractor.extract_all("Nothing happens.", [], {'won': False})
        gold = gold_predict_string(actual)
        assert 'objects_visible: no' in gold
        assert 'visible_objects: none' in gold


class TestGoldRoundTripInvariant:
    """parse(gold) → compute_reward(·, actual) ≡ 1.0"""

    def _roundtrip(self, extractor, obs, info, vocab=None, include_dev=False):
        actual = extractor.extract_all(obs, [], info)
        gold = gold_predict_string(actual, include_device_states=include_dev)
        parsed = parse_predict_block(f"<predict>{gold}</predict>", object_vocab=vocab)
        assert parsed is not None, f"gold 串解析失败: {gold}"
        predicted = prediction_to_features(parsed)
        return extractor.compute_reward(predicted, actual)

    def test_alfworld_schema(self):
        extractor = create_alfworld_schema_extractor()
        assert self._roundtrip(extractor, ALFWORLD_OBS, {'won': False}) == pytest.approx(1.0)

    def test_alfworld_won(self):
        extractor = create_alfworld_schema_extractor()
        assert self._roundtrip(extractor, ALFWORLD_OBS, {'won': True}) == pytest.approx(1.0)

    def test_hrg_schema(self):
        extractor = create_hiddenrule_schema_extractor()
        assert self._roundtrip(extractor, HRG_OBS, {'won': False},
                               vocab=HRG_OBJECT_VOCAB) == pytest.approx(1.0)

    def test_hrg_with_device_states_weighted(self):
        """C-sweep 权重下 (device_state 0.7) gold 依然满分 —— 'set to N'↔'N' 词形闭环"""
        extractor = create_hiddenrule_schema_extractor(feature_weights={
            'location_change': 0.3, 'device_state': 0.7,
            'objects_visible': 0.0, 'visible_objects': 0.0, 'task_progress': 0.0,
        })
        assert self._roundtrip(extractor, HRG_OBS, {'won': False},
                               vocab=HRG_OBJECT_VOCAB,
                               include_dev=True) == pytest.approx(1.0)

    def test_hrg_empty_room(self):
        extractor = create_hiddenrule_schema_extractor()
        obs = "You are in room 1. The room is empty."
        assert self._roundtrip(extractor, obs, {'won': False},
                               vocab=HRG_OBJECT_VOCAB) == pytest.approx(1.0)
