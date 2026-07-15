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
可验证特征提取器测试

运行: pytest tests/test_verifiable_features.py -v
"""

import pytest

from agent_system.environments.verifiable_features import (
    ALFWorldActionAvailabilityFeature,
    ALFWorldLocationChangeFeature,
    ALFWorldObjectSeenFeature,
    ALFWorldTaskProgressFeature,
    VerifiableFeature,
    create_alfworld_feature_extractor,
    parse_predict_block,
    prediction_to_features,
)

# ---------------------------------------------------------------------------
# feature_type 类属性 (compute_reward 恒 0 bug 的回归防线)
# ---------------------------------------------------------------------------

def test_feature_type_attributes():
    assert ALFWorldObjectSeenFeature.feature_type == 'object_seen'
    assert ALFWorldLocationChangeFeature.feature_type == 'location_change'
    assert ALFWorldActionAvailabilityFeature.feature_type == 'action_available'
    assert ALFWorldTaskProgressFeature.feature_type == 'task_progress'


def test_extract_feature_type_matches_class_attribute():
    extractor = ALFWorldObjectSeenFeature({'ladle'})
    feature = extractor.extract("You see a ladle.", [], {})
    assert feature.feature_type == extractor.feature_type


# ---------------------------------------------------------------------------
# ALFWorldObjectSeenFeature
# ---------------------------------------------------------------------------

class TestObjectSeenFeature:
    def setup_method(self):
        self.extractor = ALFWorldObjectSeenFeature({'ladle', 'fridge', 'knife', 'apple'})

    def test_extract_seen_objects(self):
        obs = "You open the cabinet 1. The cabinet 1 is open. In it, you see a ladle and a knife."
        feature = self.extractor.extract(obs, ['close cabinet 1'], {})
        assert 'ladle' in feature.value['seen']
        assert 'knife' in feature.value['seen']
        assert 'apple' in feature.value['not_seen']

    def test_extract_nothing_seen(self):
        obs = "You open the cabinet 1. The cabinet 1 is open. In it, you see nothing."
        feature = self.extractor.extract(obs, [], {})
        assert len(feature.value['seen']) == 0

    def test_verify_legacy_subset_form(self):
        actual = VerifiableFeature('object_seen', {'seen': ['ladle', 'knife']})
        assert self.extractor.verify(
            VerifiableFeature('object_seen', {'seen': ['ladle']}), actual) is True
        assert self.extractor.verify(
            VerifiableFeature('object_seen', {'seen': ['apple']}), actual) is False

    def test_verify_visible_bool_form(self):
        """<predict> 块的受限预测形式: {'visible': bool}"""
        seen_something = VerifiableFeature('object_seen', {'seen': ['ladle']})
        seen_nothing = VerifiableFeature('object_seen', {'seen': []})

        pred_yes = VerifiableFeature('object_seen', {'visible': True})
        pred_no = VerifiableFeature('object_seen', {'visible': False})

        assert self.extractor.verify(pred_yes, seen_something) is True
        assert self.extractor.verify(pred_yes, seen_nothing) is False
        assert self.extractor.verify(pred_no, seen_nothing) is True
        assert self.extractor.verify(pred_no, seen_something) is False

    def test_verify_type_mismatch_raises(self):
        with pytest.raises(ValueError):
            self.extractor.verify(
                VerifiableFeature('location_change', 'cabinet 1'),
                VerifiableFeature('object_seen', {'seen': []}),
            )


# ---------------------------------------------------------------------------
# ALFWorldLocationChangeFeature
# ---------------------------------------------------------------------------

class TestLocationChangeFeature:
    def setup_method(self):
        self.extractor = ALFWorldLocationChangeFeature()

    def test_extract_arrive_at(self):
        feature = self.extractor.extract("You arrive at cabinet 1. The cabinet 1 is closed.", [], {})
        assert feature.value == 'cabinet 1'

    def test_extract_go_to(self):
        feature = self.extractor.extract("You go to diningtable 1.", [], {})
        assert feature.value == 'diningtable 1'

    def test_extract_no_location(self):
        feature = self.extractor.extract("Nothing happens.", [], {})
        assert feature.value is None

    def test_verify(self):
        loc = lambda v: VerifiableFeature('location_change', v)
        assert self.extractor.verify(loc('fridge 1'), loc('fridge 1')) is True
        assert self.extractor.verify(loc('Fridge 1 '), loc('fridge 1')) is True  # 归一化
        assert self.extractor.verify(loc('fridge 1'), loc('cabinet 2')) is False
        assert self.extractor.verify(loc(None), loc(None)) is True
        assert self.extractor.verify(loc(None), loc('fridge 1')) is False
        assert self.extractor.verify(loc('fridge 1'), loc(None)) is False


# ---------------------------------------------------------------------------
# ALFWorldActionAvailabilityFeature
# ---------------------------------------------------------------------------

class TestActionAvailabilityFeature:
    def setup_method(self):
        self.extractor = ALFWorldActionAvailabilityFeature()

    def test_extract_action_patterns(self):
        actions = ['open cabinet 1', 'close cabinet 1', 'pick up ladle',
                   'put knife in fridge', 'examine cabinet 1']
        feature = self.extractor.extract("You are at cabinet 1.", actions, {})
        assert 'open' in feature.value
        assert 'pick' in feature.value
        assert 'put' in feature.value

    def test_extract_navigation_only(self):
        feature = self.extractor.extract(
            "You are at cabinet 1.", ['go to cabinet 2', 'go to fridge 1', 'look'], {})
        assert 'pick' not in feature.value

    def test_verify_subset(self):
        act = lambda v: VerifiableFeature('action_available', v)
        assert self.extractor.verify(act(['open']), act(['open', 'pick'])) is True
        assert self.extractor.verify(act(['heat']), act(['open', 'pick'])) is False


# ---------------------------------------------------------------------------
# ALFWorldTaskProgressFeature
# ---------------------------------------------------------------------------

class TestTaskProgressFeature:
    def setup_method(self):
        self.extractor = ALFWorldTaskProgressFeature()

    def test_extract(self):
        feature = self.extractor.extract(
            "Task done!", [], {'won': True, 'goal_condition_success_rate': 1.0, 'reward': 10.0})
        assert feature.value['won'] is True
        assert feature.value['success_rate'] == 1.0

    def test_verify(self):
        prog = lambda won: VerifiableFeature('task_progress', {'won': won})
        assert self.extractor.verify(prog(True), prog(True)) is True
        assert self.extractor.verify(prog(True), prog(False)) is False


# ---------------------------------------------------------------------------
# CompositeFeatureExtractor
# ---------------------------------------------------------------------------

class TestCompositeExtractor:
    def setup_method(self):
        self.composite = create_alfworld_feature_extractor(
            object_types={'ladle', 'knife', 'fridge'},
            feature_weights={
                'object_seen': 0.4,
                'location_change': 0.4,
                'action_available': 0.2,
                'task_progress': 0.0,
            }
        )
        self.obs = "You arrive at cabinet 1. The cabinet 1 is open. In it, you see a ladle."
        self.actions = ['open cabinet 1', 'pick up ladle', 'close cabinet 1']
        self.info = {'won': False, 'goal_condition_success_rate': 0.0}

    def test_extract_all_no_weight_pollution(self):
        features = self.composite.extract_all(self.obs, self.actions, self.info)
        assert set(features.keys()) == {'object_seen', 'location_change',
                                        'action_available', 'task_progress'}
        assert all(isinstance(f, VerifiableFeature) for f in features.values())

    def test_verify_all_with_parsed_prediction(self):
        actual = self.composite.extract_all(self.obs, self.actions, self.info)
        predicted = prediction_to_features(
            {'next_location': 'cabinet 1', 'target_visible': True, 'task_done': False})
        results = self.composite.verify_all(predicted, actual)
        assert results == {'location_change': True, 'object_seen': True, 'task_progress': True}

    def test_compute_reward_perfect_prediction_is_one(self):
        """回归测试: 修复前 compute_reward 因键名不匹配恒返回 0"""
        actual = self.composite.extract_all(self.obs, self.actions, self.info)
        predicted = prediction_to_features({'next_location': 'cabinet 1', 'target_visible': True})
        assert self.composite.compute_reward(predicted, actual) == pytest.approx(1.0)

    def test_compute_reward_partial_prediction(self):
        """位置错、可见性对 → 0.4*0 + 0.4*1 归一化 = 0.5"""
        actual = self.composite.extract_all(self.obs, self.actions, self.info)
        predicted = prediction_to_features({'next_location': 'fridge 1', 'target_visible': True})
        assert self.composite.compute_reward(predicted, actual) == pytest.approx(0.5)

    def test_compute_reward_only_zero_weight_feature(self):
        """只预测了权重 0 的 task_progress → 分数 0 (平凡预测不给分)"""
        actual = self.composite.extract_all(self.obs, self.actions, self.info)
        predicted = prediction_to_features({'task_done': False})
        assert self.composite.compute_reward(predicted, actual) == 0.0

    def test_compute_reward_no_prediction(self):
        actual = self.composite.extract_all(self.obs, self.actions, self.info)
        assert self.composite.compute_reward({}, actual) == 0.0

    def test_default_factory_task_progress_weight_zero(self):
        composite = create_alfworld_feature_extractor()
        weights = {e.feature_type: w for e, w in composite.extractors}
        assert weights['task_progress'] == 0.0
        assert weights['object_seen'] > 0
        assert weights['location_change'] > 0


# ---------------------------------------------------------------------------
# parse_predict_block / prediction_to_features
# ---------------------------------------------------------------------------

class TestParsePredictBlock:
    def test_full_block(self):
        text = ("<think>I should go check the cabinet.</think>\n"
                "<predict>next_location: cabinet 2; target_visible: yes; task_done: no</predict>\n"
                "<action>go to cabinet 2</action>")
        parsed = parse_predict_block(text)
        assert parsed == {'next_location': 'cabinet 2', 'target_visible': True, 'task_done': False}

    def test_missing_block_returns_none(self):
        assert parse_predict_block("<think>...</think><action>look</action>") is None
        assert parse_predict_block("") is None
        assert parse_predict_block(None) is None

    def test_empty_block_returns_none(self):
        assert parse_predict_block("<predict></predict>") is None
        assert parse_predict_block("<predict>gibberish without colon</predict>") is None

    def test_partial_fields(self):
        parsed = parse_predict_block("<predict>next_location: fridge 1</predict>")
        assert parsed == {'next_location': 'fridge 1'}

    def test_none_location(self):
        parsed = parse_predict_block("<predict>next_location: none; target_visible: no</predict>")
        assert parsed['next_location'] is None
        assert parsed['target_visible'] is False

    def test_unparseable_bool_is_omitted(self):
        parsed = parse_predict_block(
            "<predict>target_visible: maybe; task_done: no</predict>")
        assert 'target_visible' not in parsed
        assert parsed['task_done'] is False

    def test_case_insensitive_and_multiline(self):
        text = "<PREDICT>\nNext_Location: Cabinet 3;\nTarget_Visible: YES\n</PREDICT>"
        parsed = parse_predict_block(text)
        assert parsed == {'next_location': 'cabinet 3', 'target_visible': True}

    def test_unknown_keys_ignored(self):
        parsed = parse_predict_block(
            "<predict>mood: happy; next_location: drawer 5</predict>")
        assert parsed == {'next_location': 'drawer 5'}


class TestPredictionToFeatures:
    def test_full_mapping(self):
        features = prediction_to_features(
            {'next_location': 'cabinet 2', 'target_visible': True, 'task_done': False})
        assert features['location_change'].value == 'cabinet 2'
        assert features['object_seen'].value == {'visible': True}
        assert features['task_progress'].value == {'won': False}

    def test_omitted_fields_produce_no_features(self):
        features = prediction_to_features({'next_location': None})
        assert set(features.keys()) == {'location_change'}
        assert features['location_change'].value is None
