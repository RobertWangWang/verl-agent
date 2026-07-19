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

"""C-sweep 接线测试 (coverage_level → Φ mask, docs/hiddenrule_gym_design.md §2.1)"""

import pytest

from agent_system.environments.env_package.hiddenrule.features import (
    HRGDeviceStateFeature,
    PhiMaskedExtractor,
    apply_phi_mask,
    create_hiddenrule_schema_extractor,
)
from agent_system.environments.env_package.hiddenrule.hiddenrule.coverage import (
    calibrate_masks,
    coverage,
    enumerate_reachable,
    sweep_fields,
)
from agent_system.environments.env_package.hiddenrule.hiddenrule.env import HiddenRuleEnv
from agent_system.environments.env_package.hiddenrule.hiddenrule.world import (
    HRGConfig,
    generate_world,
)
from agent_system.environments.verifiable_features import (
    VerifiableFeature,
    parse_predict_block,
    prediction_to_features,
)


def _conj_config(**kwargs) -> HRGConfig:
    defaults = dict(rule_families=['conj'], n_rooms=4, n_devices=4)
    defaults.update(kwargs)
    return HRGConfig(**defaults)


# ---------------------------------------------------------------------------
# 解析器: device_states 字段
# ---------------------------------------------------------------------------

class TestParseDeviceStates:
    def test_parse_basic(self):
        parsed = parse_predict_block(
            "<predict>next_location: room 2; device_states: lever_a=up, dial_b=2; "
            "task_done: no</predict>")
        assert parsed['device_states'] == [('dial_b', 'set to 2'), ('lever_a', 'up')]

    def test_parse_none(self):
        parsed = parse_predict_block("<predict>device_states: none</predict>")
        assert parsed['device_states'] == []

    def test_parse_set_to_form(self):
        parsed = parse_predict_block("<predict>device_states: dial_c set to 3</predict>")
        assert parsed['device_states'] == [('dial_c', 'set to 3')]

    def test_parse_garbage_dropped(self):
        parsed = parse_predict_block(
            "<predict>device_states: teapot=hot, lever_b=down</predict>")
        assert parsed['device_states'] == [('lever_b', 'down')]

    def test_prediction_to_features(self):
        parsed = {'device_states': [('lever_a', 'up')]}
        feats = prediction_to_features(parsed)
        assert feats['device_state'].feature_type == 'device_state'
        assert feats['device_state'].value['pairs'] == [('lever_a', 'up')]


# ---------------------------------------------------------------------------
# device_state 的 F1 部分分
# ---------------------------------------------------------------------------

class TestDeviceStateF1:
    def _feat(self, pairs):
        return VerifiableFeature(feature_type='device_state', value={'pairs': pairs})

    def test_exact_match(self):
        f = HRGDeviceStateFeature()
        pairs = [('lever_a', 'up'), ('dial_b', 'set to 2')]
        assert f.verify_score(self._feat(pairs), self._feat(pairs)) == 1.0

    def test_partial(self):
        f = HRGDeviceStateFeature()
        score = f.verify_score(
            self._feat([('lever_a', 'up')]),
            self._feat([('lever_a', 'up'), ('dial_b', 'set to 2')]))
        assert 0.0 < score < 1.0  # precision 1, recall 0.5 → F1 2/3

    def test_empty_empty(self):
        f = HRGDeviceStateFeature()
        assert f.verify_score(self._feat([]), self._feat([])) == 1.0

    def test_wrong_state(self):
        f = HRGDeviceStateFeature()
        assert f.verify_score(
            self._feat([('lever_a', 'up')]), self._feat([('lever_a', 'down')])) == 0.0


# ---------------------------------------------------------------------------
# apply_phi_mask / PhiMaskedExtractor
# ---------------------------------------------------------------------------

class TestPhiMask:
    def _features(self):
        return {
            'location_change': VerifiableFeature('location_change', 'room 2'),
            'device_state': VerifiableFeature(
                'device_state',
                {'pairs': [('lever_a', 'up'), ('dial_b', 'set to 2')]}),
            'objects_visible': VerifiableFeature('objects_visible', {'seen': ['note_1']}),
        }

    def test_room_dropped(self):
        out = apply_phi_mask(self._features(), ['device:lever_a'])
        assert 'location_change' not in out
        assert out['device_state'].value['pairs'] == [('lever_a', 'up')]

    def test_device_filtered(self):
        out = apply_phi_mask(self._features(), ['room', 'device:dial_b'])
        assert 'location_change' in out
        assert out['device_state'].value['pairs'] == [('dial_b', 'set to 2')]

    def test_no_device_fields_drops_feature(self):
        out = apply_phi_mask(self._features(), ['room'])
        assert 'device_state' not in out

    def test_unmanaged_features_pass_through(self):
        out = apply_phi_mask(self._features(), ['room'])
        assert 'objects_visible' in out

    def test_masked_extractor_renormalizes(self):
        """mask 丢掉 location 后, 只对 device_state 计分 (权重重归一, 满分仍可达 1)"""
        base = create_hiddenrule_schema_extractor(feature_weights={
            'location_change': 0.3, 'device_state': 0.7,
            'objects_visible': 0.0, 'visible_objects': 0.0, 'task_progress': 0.0,
        })
        masked = PhiMaskedExtractor(base, ['device:lever_a'])
        predicted = {
            'location_change': VerifiableFeature('location_change', 'room 9'),  # 错的, 但会被 mask 掉
            'device_state': VerifiableFeature('device_state', {'pairs': [('lever_a', 'up')]}),
        }
        actual = {
            'location_change': VerifiableFeature('location_change', 'room 2'),
            'device_state': VerifiableFeature('device_state', {'pairs': [('lever_a', 'up')]}),
        }
        assert masked.compute_reward(predicted, actual) == pytest.approx(1.0)
        # 同输入不 mask: location 错 → 0.3 权重丢分
        assert base.compute_reward(predicted, actual) == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# calibrate_masks 的字段池限定 + env 接线
# ---------------------------------------------------------------------------

class TestSweepWiring:
    def test_sweep_fields_no_doors(self):
        world = generate_world(_conj_config(key_door_prob=1.0), seed=7)
        fields = sweep_fields(world)
        assert 'room' in fields
        assert all(not f.startswith('door:') for f in fields)

    def test_calibrate_with_field_pool(self):
        world = generate_world(_conj_config(), seed=3)
        states = enumerate_reachable(world)
        pool = sweep_fields(world)
        ladder = calibrate_masks(world, targets=(0.4,), states=states, fields=pool)
        mask = ladder[0.4]
        assert mask <= frozenset(pool)
        assert coverage(world, mask, states=states) >= 0.4 - 1e-9

    def test_env_exposes_mask_and_coverage(self):
        env = HiddenRuleEnv(_conj_config(coverage_level=0.4))
        _, info = env.reset(seed=11)
        assert info['phi_mask'] is not None
        assert info['phi_coverage'] >= 0.4 - 1e-9
        # mask 恒定贯穿 episode
        obs, r, done, info2 = env.step('look')
        assert info2['phi_mask'] == info['phi_mask']
        assert info2['phi_coverage'] == info['phi_coverage']

    def test_env_full_coverage_zero_cost_path(self):
        env = HiddenRuleEnv(_conj_config(coverage_level=1.0))
        _, info = env.reset(seed=11)
        assert info['phi_mask'] is None
        assert info['phi_coverage'] is None

    def test_env_mask_deterministic(self):
        env1 = HiddenRuleEnv(_conj_config(coverage_level=0.6))
        env2 = HiddenRuleEnv(_conj_config(coverage_level=0.6))
        _, i1 = env1.reset(seed=23)
        HiddenRuleEnv._coverage_cache.clear()  # 击穿缓存, 验证重算一致
        _, i2 = env2.reset(seed=23)
        assert i1['phi_mask'] == i2['phi_mask']
        assert i1['phi_coverage'] == i2['phi_coverage']


# ---------------------------------------------------------------------------
# 臂 D: vault_openable 上界 Φ (特权 latent 验证)
# ---------------------------------------------------------------------------

class TestVaultOpenableFeature:
    def test_extract_from_privileged_latent(self):
        from agent_system.environments.env_package.hiddenrule.features import HRGVaultOpenableFeature
        f = HRGVaultOpenableFeature()
        feat = f.extract("You are in room 1.", [], {'latent_state': {'vault_openable': True}})
        assert feat.value == {'openable': True}
        feat2 = f.extract("You are in room 1.", [], {'latent_state': {'vault_openable': False}})
        assert feat2.value == {'openable': False}
        # 缺 latent 时安全回落 False
        assert f.extract("x", [], {}).value == {'openable': False}

    def test_parse_and_roundtrip(self):
        from agent_system.environments.env_package.hiddenrule.features import (
            HRG_OBJECT_VOCAB, create_hiddenrule_schema_extractor)
        from agent_system.environments.verifiable_features import (
            parse_predict_block, prediction_to_features)
        parsed = parse_predict_block(
            "<predict>next_location: room 2; vault_openable: yes; task_done: no</predict>",
            object_vocab=HRG_OBJECT_VOCAB)
        assert parsed['vault_openable'] is True
        feats = prediction_to_features(parsed)
        assert feats['vault_openable'].value == {'openable': True}
        # 臂 D 权重下满分闭环: 预测与特权 latent 一致 → 1.0
        ex = create_hiddenrule_schema_extractor(feature_weights={
            'location_change': 0.3, 'vault_openable': 0.7,
            'objects_visible': 0.0, 'visible_objects': 0.0,
            'device_state': 0.0, 'task_progress': 0.0})
        actual = ex.extract_all("You are in room 2.", [],
                                {'latent_state': {'vault_openable': True}, 'won': False})
        assert ex.compute_reward(feats, actual) == pytest.approx(1.0)
        # 预测错 → 只得 location 份额 0.3
        parsed_no = parse_predict_block(
            "<predict>next_location: room 2; vault_openable: no</predict>",
            object_vocab=HRG_OBJECT_VOCAB)
        assert ex.compute_reward(prediction_to_features(parsed_no), actual) == pytest.approx(0.3)

    def test_env_end_to_end_latent_matches_open_vault(self):
        """特权 latent 与环境真实可开性一致 (oracle 终点即 vault_openable 为真的状态)"""
        from agent_system.environments.env_package.hiddenrule.hiddenrule.env import HiddenRuleEnv
        from agent_system.environments.env_package.hiddenrule.hiddenrule.oracle import solve
        from agent_system.environments.env_package.hiddenrule.hiddenrule.world import HRGConfig, generate_world
        cfg = HRGConfig(rule_families=['conj'], n_rooms=4, n_devices=4)
        env = HiddenRuleEnv(cfg)
        obs, info = env.reset(seed=5)
        assert info['latent_state']['vault_openable'] in (True, False)
        solution = solve(env.world)
        assert solution is not None
        done = False
        for action in solution:
            if action == 'open vault':
                assert env._info(True)['latent_state']['vault_openable'] is True
            obs, r, done, info = env.step(action)
        assert info['won'] is True
