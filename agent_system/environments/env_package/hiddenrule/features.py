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
HiddenRule-Gym 的 schema 级可验证特征 (v0.2 协议; docs/hiddenrule_gym_design.md §4)

设计: 复用 ALFWorld 的 feature_type 命名 (location_change / objects_visible /
visible_objects / task_progress),只有 extract() 的解析按 HRG 观测格式实现
—— PS 管线其余部分 (parse_predict_block / PredictiveMemory / rollout / trainer
注入) 零改动跨环境复用,同时验证 S4 特征协议的可移植性。

词表 = 环境本体 (机关命名方案 + 便签/钥匙名),任务无关。
"""

import re
import string
from typing import Any, Dict, List, Set

from agent_system.environments.verifiable_features import (
    BaseFeatureExtractor,
    CompositeFeatureExtractor,
    VerifiableFeature,
)

# schema 级词表: 生成器的命名方案决定的封闭集合 (与具体 episode/任务无关)
HRG_OBJECT_VOCAB: Set[str] = (
    {f"{kind}_{letter.lower()}" for kind in ("lever", "dial", "button")
     for letter in string.ascii_uppercase[:8]}
    | {f"note_{i}" for i in range(1, 5)}
    | {"brass_key"}
)

_ROOM_RE = re.compile(r'you are in room (\d+)', re.IGNORECASE)
_DEVICE_RE = re.compile(r'\b((?:lever|dial|button)_[a-h])\b', re.IGNORECASE)
_SEE_LINE_RE = re.compile(r'you see:? (.*?)(?:\.|$)', re.IGNORECASE | re.MULTILINE)


class HRGLocationFeature(BaseFeatureExtractor):
    """agent 所在房间 ("You are in room 3 of 5" → 'room 3')"""

    feature_type = 'location_change'

    def extract(self, observation: str, admissible_actions: List[str], info: Dict[str, Any]) -> VerifiableFeature:
        match = _ROOM_RE.search(observation)
        value = f"room {match.group(1)}" if match else None
        return VerifiableFeature(feature_type=self.feature_type, value=value)

    def verify(self, predicted: VerifiableFeature, actual: VerifiableFeature) -> bool:
        if predicted.feature_type != self.feature_type or actual.feature_type != self.feature_type:
            raise ValueError("Feature type mismatch")
        pred, act = predicted.value, actual.value
        if pred is None and act is None:
            return True
        if pred is None or act is None:
            return False
        return pred.lower().strip() == act.lower().strip()


def _visible_names(observation: str) -> Set[str]:
    """观测中可见的机关/便签/物品名 (小写规范)"""
    names = {m.lower() for m in _DEVICE_RE.findall(observation)}
    for match in _SEE_LINE_RE.finditer(observation.lower()):
        names.update(t for t in re.findall(r'[a-z][a-z_0-9]*', match.group(1))
                     if t in HRG_OBJECT_VOCAB)
    return names


class HRGObjectsVisibleFeature(BaseFeatureExtractor):
    """布尔: 下一观测是否列出至少一个机关/物品 (对应 predict 块 objects_visible)"""

    feature_type = 'objects_visible'

    def extract(self, observation: str, admissible_actions: List[str], info: Dict[str, Any]) -> VerifiableFeature:
        return VerifiableFeature(feature_type=self.feature_type,
                                 value={'seen': sorted(_visible_names(observation))})

    def verify(self, predicted: VerifiableFeature, actual: VerifiableFeature) -> bool:
        if predicted.feature_type != self.feature_type or actual.feature_type != self.feature_type:
            raise ValueError("Feature type mismatch")
        return bool(predicted.value.get('visible')) == bool(actual.value.get('seen'))


class HRGVisibleObjectsF1Feature(BaseFeatureExtractor):
    """开放集: 预测将看到哪些机关/物品名,F1 部分分 (权重 0 仅日志,同 ALFWorld 探针)"""

    feature_type = 'visible_objects'

    def extract(self, observation: str, admissible_actions: List[str], info: Dict[str, Any]) -> VerifiableFeature:
        return VerifiableFeature(feature_type=self.feature_type,
                                 value={'objects': sorted(_visible_names(observation))})

    def verify(self, predicted: VerifiableFeature, actual: VerifiableFeature) -> bool:
        return self.verify_score(predicted, actual) == 1.0

    def verify_score(self, predicted: VerifiableFeature, actual: VerifiableFeature) -> float:
        if predicted.feature_type != self.feature_type or actual.feature_type != self.feature_type:
            raise ValueError("Feature type mismatch")
        pred = set(predicted.value.get('objects', []))
        act = set(actual.value.get('objects', []))
        if not pred and not act:
            return 1.0
        if not pred or not act:
            return 0.0
        tp = len(pred & act)
        precision, recall = tp / len(pred), tp / len(act)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


class HRGDeviceStateFeature(BaseFeatureExtractor):
    """(device, state) 二元组集合,C-sweep 的主预测目标 (predict 块 device_states 字段;
    对应 ALFWorld 的 receptacle_state 槽位)。F1 部分分,与 visible_objects 同风格。"""

    feature_type = 'device_state'

    STATE_RE = re.compile(r'\b((?:lever|dial)_[a-h]) is (up|down|set to \d+)', re.IGNORECASE)

    def extract(self, observation: str, admissible_actions: List[str], info: Dict[str, Any]) -> VerifiableFeature:
        pairs = sorted({(m.group(1).lower(), m.group(2).lower())
                        for m in self.STATE_RE.finditer(observation)})
        return VerifiableFeature(feature_type=self.feature_type, value={'pairs': pairs})

    def verify(self, predicted: VerifiableFeature, actual: VerifiableFeature) -> bool:
        if predicted.feature_type != self.feature_type or actual.feature_type != self.feature_type:
            raise ValueError("Feature type mismatch")
        return set(map(tuple, predicted.value.get('pairs', []))) <= \
            set(map(tuple, actual.value.get('pairs', [])))

    def verify_score(self, predicted: VerifiableFeature, actual: VerifiableFeature) -> float:
        if predicted.feature_type != self.feature_type or actual.feature_type != self.feature_type:
            raise ValueError("Feature type mismatch")
        pred = set(map(tuple, predicted.value.get('pairs', [])))
        act = set(map(tuple, actual.value.get('pairs', [])))
        if not pred and not act:
            return 1.0
        if not pred or not act:
            return 0.0
        tp = len(pred & act)
        precision, recall = tp / len(pred), tp / len(act)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


class HRGTaskProgressFeature(BaseFeatureExtractor):
    """vault 是否已开 (info['won']);对应 predict 块 task_done,权重 0 仅日志"""

    feature_type = 'task_progress'

    def extract(self, observation: str, admissible_actions: List[str], info: Dict[str, Any]) -> VerifiableFeature:
        return VerifiableFeature(feature_type=self.feature_type,
                                 value={'won': bool(info.get('won', False))})

    def verify(self, predicted: VerifiableFeature, actual: VerifiableFeature) -> bool:
        if predicted.feature_type != self.feature_type or actual.feature_type != self.feature_type:
            raise ValueError("Feature type mismatch")
        return predicted.value.get('won', False) == actual.value.get('won', False)


def apply_phi_mask(features: Dict[str, VerifiableFeature],
                   mask: List[str]) -> Dict[str, VerifiableFeature]:
    """
    按覆盖度字段 mask 裁剪特征字典 (C-sweep, design doc §2.1)。
    mask 字段词表来自 coverage.sweep_fields: 'room' / 'device:<name>'。

    - 'room' 不在 mask → 丢弃 location_change;
    - device_state 的 pairs 过滤到 mask 内的机关名; mask 无任何 device 字段 → 整个丢弃;
    - 其余特征 (objects_visible / visible_objects 探针 / task_progress) 不受 mask 管辖,
      由 feature_weights 治理 —— C-sweep 臂必须把它们的权重设 0 (探针除外,本就 0),
      否则测得的 C 与被奖励的 Φ 家族不一致。
    """
    allowed_devices = {f.split(':', 1)[1] for f in mask if f.startswith('device:')}
    out: Dict[str, VerifiableFeature] = {}
    for ftype, feat in features.items():
        if ftype == 'location_change':
            if 'room' in mask:
                out[ftype] = feat
        elif ftype == 'device_state':
            if allowed_devices:
                pairs = [p for p in feat.value.get('pairs', [])
                         if p[0] in allowed_devices]
                out[ftype] = VerifiableFeature(feature_type=ftype, value={'pairs': pairs})
        else:
            out[ftype] = feat
    return out


class PhiMaskedExtractor:
    """
    组合提取器的 Φ-mask 视图: extract/verify/compute 全部委托底层提取器,
    仅在验证与计分前对 predicted 与 actual 两侧同步施加 apply_phi_mask。
    每 env 每 episode 的 mask 恒定 (world 级),包装对象可即用即建。
    """

    def __init__(self, base: CompositeFeatureExtractor, mask: List[str]):
        self._base = base
        self._mask = mask

    @property
    def extractors(self):
        return self._base.extractors

    def extract_all(self, observation, admissible_actions, info):
        return apply_phi_mask(
            self._base.extract_all(observation, admissible_actions, info), self._mask)

    def verify_all(self, predicted, actual):
        return self._base.verify_all(apply_phi_mask(predicted, self._mask), actual)

    def compute_reward(self, predicted, actual):
        return self._base.compute_reward(apply_phi_mask(predicted, self._mask), actual)


def create_hiddenrule_schema_extractor(feature_weights: Dict[str, float] = None) -> CompositeFeatureExtractor:
    """HiddenRule-Gym 的 schema 级组合提取器 (任务无关,单实例全环境共享)"""
    if feature_weights is None:
        feature_weights = {
            'location_change': 0.5,
            'objects_visible': 0.5,
            'visible_objects': 0.0,  # F1 探针仅日志
            'device_state': 0.0,     # 不被预测,进特征池
            'task_progress': 0.0,    # 平凡预测仅日志
        }
    extractors = [
        (HRGLocationFeature(), feature_weights.get('location_change', 0.5)),
        (HRGObjectsVisibleFeature(), feature_weights.get('objects_visible', 0.5)),
        (HRGVisibleObjectsF1Feature(), feature_weights.get('visible_objects', 0.0)),
        (HRGDeviceStateFeature(), feature_weights.get('device_state', 0.0)),
        (HRGTaskProgressFeature(), feature_weights.get('task_progress', 0.0)),
    ]
    return CompositeFeatureExtractor(extractors)
