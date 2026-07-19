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

"""锚点 QA 对照臂 (docs/ps_grpo_integration_design.md §9): 解析/评分/recorder"""

import pytest

from agent_system.memory.anchor_qa import AnchorQARecorder
from agent_system.environments.verifiable_features import (
    create_alfworld_schema_extractor,
    parse_recall_block,
)

OBS_STEP1 = "You arrive at cabinet 2. In it, you see a mug 1, and a plate 2."
OBS_STEP2 = "You pick up the mug 1 from the cabinet 2."
OBS_STEP3 = "You arrive at fridge 1. The fridge 1 is closed."


class TestParseRecall:
    def test_basic(self):
        parsed = parse_recall_block("<recall>location: cabinet 2; objects: mug, plate</recall>")
        assert parsed['location'] == 'cabinet 2'
        assert parsed['objects'] == ['mug', 'plate']

    def test_none_values(self):
        parsed = parse_recall_block("<recall>location: none; objects: none</recall>")
        assert parsed['location'] is None
        assert parsed['objects'] == []

    def test_vocab_filter(self):
        parsed = parse_recall_block("<recall>objects: mug, unicorn</recall>")
        assert parsed['objects'] == ['mug']

    def test_missing_block(self):
        assert parse_recall_block("<action>go</action>") is None
        assert parse_recall_block("") is None


class TestRecorder:
    def _recorder(self, lag=2):
        r = AnchorQARecorder(create_alfworld_schema_extractor(), lag=lag)
        r.reset(batch_size=2)
        r.record(0, 1, OBS_STEP1, [], {'won': False})
        r.record(0, 2, OBS_STEP2, [], {'won': False})
        r.record(0, 3, OBS_STEP3, [], {'won': False})
        return r

    def test_anchor_step(self):
        r = self._recorder(lag=2)
        assert r.anchor_step(1) is None
        assert r.anchor_step(2) is None
        assert r.anchor_step(3) == 1
        assert r.anchor_step(5) == 3

    def test_perfect_recall(self):
        r = self._recorder()
        parsed = parse_recall_block("<recall>location: cabinet 2; objects: mug, plate</recall>")
        # 存档 ground truth = extractor 对 OBS_STEP1 的输出 (含容器名 cabinet)
        assert r.score(0, 1, parsed) == pytest.approx(1.0)

    def test_wrong_location_half_score(self):
        r = self._recorder()
        parsed = parse_recall_block("<recall>location: fridge 1; objects: mug, plate</recall>")
        assert r.score(0, 1, parsed) == pytest.approx(0.5)

    def test_partial_objects_f1(self):
        r = self._recorder()
        parsed = parse_recall_block("<recall>location: cabinet 2; objects: mug</recall>")
        s = r.score(0, 1, parsed)
        assert 0.5 < s < 1.0  # 位置满分 + F1 部分分

    def test_parse_failure_zero(self):
        r = self._recorder()
        assert r.score(0, 1, None) == 0.0

    def test_missing_anchor_zero(self):
        r = self._recorder()
        parsed = parse_recall_block("<recall>location: cabinet 2</recall>")
        assert r.score(1, 1, parsed) == 0.0  # env 1 无存档

    def test_no_location_obs_recalls_none(self):
        r = self._recorder()
        # OBS_STEP2 无位置行且无可见物列表 → 正确回忆是双 'none'
        parsed = parse_recall_block("<recall>location: none; objects: none</recall>")
        assert r.score(0, 2, parsed) == pytest.approx(1.0)

    def test_ground_truth_shares_extractor_with_ps(self):
        """同源性: 存档真值 = schema extractor 输出,与 PS 验证器一致"""
        extractor = create_alfworld_schema_extractor()
        feats = extractor.extract_all(OBS_STEP1, [], {'won': False})
        truth_objects = set(feats['visible_objects'].value['objects'])
        r = self._recorder()
        parsed = {'objects': sorted(truth_objects)}
        assert r.score(0, 1, parsed) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# R13 自报告臂 (设计 §10): 解析
# ---------------------------------------------------------------------------

class TestParseReport:
    def test_integer_percent(self):
        from agent_system.environments.verifiable_features import parse_report_block
        assert parse_report_block("<report>confidence: 85</report>") == pytest.approx(0.85)
        assert parse_report_block("<report>confidence: 85%</report>") == pytest.approx(0.85)

    def test_fraction_form(self):
        from agent_system.environments.verifiable_features import parse_report_block
        assert parse_report_block("<report>confidence: 0.4</report>") == pytest.approx(0.4)

    def test_clamped(self):
        from agent_system.environments.verifiable_features import parse_report_block
        assert parse_report_block("<report>confidence: 250</report>") == 1.0

    def test_missing(self):
        from agent_system.environments.verifiable_features import parse_report_block
        assert parse_report_block("<action>go</action>") is None
        assert parse_report_block("<report>very sure</report>") is None
        assert parse_report_block("") is None

    def test_manager_imports_resolve(self):
        """回归: manager 引用的三个解析器都必须真的被 import (efc33ef 的潜伏 NameError)"""
        import agent_system.environments.env_manager as m
        assert callable(m.parse_recall_block)
        assert callable(m.parse_report_block)
        assert callable(m.parse_predict_block)
