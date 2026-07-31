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

"""ScienceWorld 接线 (SW01/02): Φ 提取器 / 解析 / gold 回环 / projection / manager。

观测串取自 2026-07-31 E 机真机 probe (boil, variation 0), 非手造。
"""

import numpy as np

from agent_system.environments.env_package.sciworld.projection import sciworld_projection
from agent_system.environments.verifiable_features import (
    SCIWORLD_OBJECT_VOCAB,
    create_sciworld_feature_extractor,
    gold_predict_string,
    parse_predict_block,
    prediction_to_features,
)

# —— E 机真机观测 (probe 2026-07-31) ——
LOOK_HALLWAY = (
    "This room is called the hallway. In it, you see: \n\tthe agent\n\ta substance "
    "called air\n\ta picture\nYou also see:\n\tA door to the art studio (that is "
    "closed)\n\tA door to the kitchen (that is closed)"
)
LOOK_KITCHEN = (
    "This room is called the kitchen. In it, you see: \n\tthe agent\n\ta substance "
    "called air\n\ta counter\n\ta sink\n\ta stove\n\ta thermometer\n\tan apple\n"
    "You also see:\n\tA door to the hallway (that is open)"
)
MOVE_OBS = "You move to the kitchen."
OPEN_OBS = "The door is now open."


class TestSciWorldFeatures:
    def setup_method(self):
        self.ext = create_sciworld_feature_extractor()

    def test_location_from_move_obs(self):
        feats = self.ext.extract_all(MOVE_OBS, [], {'look': LOOK_KITCHEN})
        assert feats['location_change'].value == 'kitchen'

    def test_location_none_when_no_move(self):
        feats = self.ext.extract_all(OPEN_OBS, [], {'look': LOOK_HALLWAY})
        assert feats['location_change'].value is None

    def test_visible_objects_from_look_multiline(self):
        # ALFWorld 的单行正则对 SW 多行列表失效 —— 这是 SW 专用行段解析的回归锚
        feats = self.ext.extract_all(MOVE_OBS, [], {'look': LOOK_KITCHEN})
        objs = set(feats['visible_objects'].value['objects'])
        assert {'counter', 'sink', 'stove', 'thermometer', 'apple'} <= objs
        assert 'agent' not in objs
        # "You also see:" 之后的门列表不进物体集
        assert 'door' not in objs

    def test_objects_visible_bool(self):
        feats = self.ext.extract_all(MOVE_OBS, [], {'look': LOOK_KITCHEN})
        assert feats['objects_visible'].value['seen']
        empty_look = "This room is called the void. In it, you see: \n\tthe agent\n"
        feats2 = self.ext.extract_all(OPEN_OBS, [], {'look': empty_look})
        assert not feats2['objects_visible'].value['seen']

    def test_gold_roundtrip_reward_one(self):
        """一致性不变量 (S6 同款): parse(gold) 对同一 actual 计满分。"""
        info = {'look': LOOK_KITCHEN}
        actual = self.ext.extract_all(MOVE_OBS, [], info)
        gold = gold_predict_string(actual)
        parsed = parse_predict_block(f"<predict>{gold}</predict>",
                                     object_vocab=SCIWORLD_OBJECT_VOCAB)
        assert parsed is not None
        predicted = prediction_to_features(parsed)
        score = self.ext.compute_reward(predicted, actual)
        assert abs(score - 1.0) < 1e-9

    def test_wrong_room_prediction_penalized(self):
        info = {'look': LOOK_KITCHEN}
        actual = self.ext.extract_all(MOVE_OBS, [], info)
        parsed = parse_predict_block(
            "<predict>next_location: greenhouse; objects_visible: yes; "
            "visible_objects: apple</predict>",
            object_vocab=SCIWORLD_OBJECT_VOCAB)
        predicted = prediction_to_features(parsed)
        score = self.ext.compute_reward(predicted, actual)
        assert score < 1.0


class TestSciWorldProjection:
    def test_extracts_action(self):
        acts, valids = sciworld_projection(
            ["<think>need heat</think><action>Activate Stove</action>"],
            require_think=True)
        assert acts == ['activate stove'] and valids == [1]

    def test_no_think_invalid_when_required(self):
        acts, valids = sciworld_projection(
            ["<action>go to kitchen</action>"], require_think=True)
        assert valids == [0] and acts == ['go to kitchen']

    def test_qwen3_mode_no_think_ok(self):
        acts, valids = sciworld_projection(
            ["<action>go to kitchen</action>"], require_think=False)
        assert valids == [1]

    def test_missing_action_tag_invalid(self):
        acts, valids = sciworld_projection(["just rambling text"], require_think=False)
        assert valids == [0]


class TestSciWorldPredictParse:
    def test_template_format_parses(self):
        # 与 prompts/sciworld.py 的指令格式逐字段对应
        txt = ("<predict>next_location: none; objects_visible: yes; "
               "visible_objects: thermometer, stove</predict>")
        parsed = parse_predict_block(txt, object_vocab=SCIWORLD_OBJECT_VOCAB)
        assert parsed['next_location'] is None
        assert parsed['objects_visible'] is True
        assert set(parsed['visible_objects']) == {'thermometer', 'stove'}

    def test_vocab_filters_noise(self):
        txt = "<predict>visible_objects: flurbwump, stove</predict>"
        parsed = parse_predict_block(txt, object_vocab=SCIWORLD_OBJECT_VOCAB)
        assert parsed['visible_objects'] == ['stove']


class _FakeEnvs:
    """鸭子类型 SciWorldHTTPEnv: 两 env, 单步后一动一静。"""

    num_envs = 2

    def __init__(self):
        self._infos = [
            dict(look=LOOK_HALLWAY, inventory='In your inventory, you see:\n\tan orange\n',
                 available_actions=['go OBJ', 'open OBJ', 'activate OBJ'],
                 possible_objects=['kitchen', 'door to kitchen'],
                 task_description='Your task is to boil water.',
                 task_score=0.0, variation_idx=0, won=False),
            dict(look=LOOK_HALLWAY, inventory='',
                 available_actions=['go OBJ', 'open OBJ'],
                 possible_objects=['kitchen'],
                 task_description='Your task is to boil water.',
                 task_score=0.0, variation_idx=3, won=False),
        ]

    def reset(self):
        return [LOOK_HALLWAY, LOOK_HALLWAY], [dict(i) for i in self._infos]

    def step(self, actions):
        assert len(actions) == 2
        infos = [dict(self._infos[0], look=LOOK_KITCHEN),
                 dict(self._infos[1])]
        return [MOVE_OBS, OPEN_OBS], [0.0, 0.0], [False, False], infos

    def close(self):
        pass


def _make_manager(pred_enabled: bool):
    from functools import partial

    from omegaconf import OmegaConf

    from agent_system.environments.env_manager import SciWorldEnvironmentManager

    config = OmegaConf.create({
        'env': {
            'history_length': 2,
            'sciworld': {
                'require_think_tags': False,
                'prediction': {'enable': pred_enabled, 'horizon': 1,
                               'reward_mode': 'potential', 'collect_gold': pred_enabled,
                               'feature_weights': None},
            },
        },
    })
    projection_f = partial(sciworld_projection, require_think=False)
    return SciWorldEnvironmentManager(_FakeEnvs(), projection_f, config)


class TestSciWorldManager:
    def test_reset_and_prompt_composition(self):
        mgr = _make_manager(pred_enabled=False)
        obs, infos = mgr.reset({})
        assert len(obs['text']) == 2
        t = obs['text'][0]
        assert 'boil water' in t
        assert 'hallway' in t
        assert "go OBJ" in t
        assert 'Objects you can refer to' in t
        assert '<predict>' not in t  # 基线模板无 predict 指令

    def test_ps_arm_prompt_and_reward_channel(self):
        mgr = _make_manager(pred_enabled=True)
        obs, _ = mgr.reset({})
        assert '<predict>' in obs['text'][0]
        resp = ("<predict>next_location: kitchen; objects_visible: yes; "
                "visible_objects: stove, apple</predict>"
                "<action>go to kitchen</action>")
        resp2 = ("<predict>next_location: none; objects_visible: yes; "
                 "visible_objects: picture</predict>"
                 "<action>open door to kitchen</action>")
        next_obs, rewards, dones, infos = mgr.step([resp, resp2])
        assert infos[0]['pred_parse_valid'] and infos[1]['pred_parse_valid']
        # env0 预测 kitchen 且真移动到 kitchen → location 命中
        assert infos[0]['pred_accuracy'] > 0.4
        # gold 采集在位且可回环
        assert 'gold_predict' in infos[0]
        parsed = parse_predict_block(f"<predict>{infos[0]['gold_predict']}</predict>",
                                     object_vocab=SCIWORLD_OBJECT_VOCAB)
        assert parsed is not None
        # 组装观测把 step obs 与新 look 都带上
        assert 'You move to the kitchen.' in mgr.pre_text_obs[0]
        assert 'This room is called the kitchen' in mgr.pre_text_obs[0]

    def test_success_metric_binarized(self):
        mgr = _make_manager(pred_enabled=False)
        mgr.reset({})
        success = {'success_rate': [], 'sciworld_task_score (not success_rate)': []}
        total_batch_list = [[{'active_masks': True}]]
        total_infos = [[{'won': True, 'task_score': 100.0}]]
        mgr._process_batch(0, total_batch_list, total_infos, success)
        assert success['success_rate'] == [1.0]
        assert success['sciworld_task_score (not success_rate)'] == [100.0]

    def test_action_valid_flag(self):
        mgr = _make_manager(pred_enabled=False)
        mgr.reset({})
        _, _, _, infos = mgr.step(["<action>go to kitchen</action>", "gibberish"])
        assert bool(np.asarray(infos[0]['is_action_valid'])) is True
        assert bool(np.asarray(infos[1]['is_action_valid'])) is False
