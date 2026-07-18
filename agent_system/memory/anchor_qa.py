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
锚点 QA 对照臂的观测存档与评分 (docs/ps_grpo_integration_design.md §9)

每步用与 PS 同一套 schema extractors 把观测特征存档;回忆答案按
w_loc·位置精确匹配 + w_obj·可见物 F1 评分。ground truth 与 PS 严格同源。
"""

from typing import Any, Dict, List, Optional


class AnchorQARecorder:
    """按 (env_idx, step) 存档观测特征并对 <recall> 答案评分。step 从 1 起。"""

    def __init__(self, feature_extractor, lag: int = 2,
                 weight_location: float = 0.5, weight_objects: float = 0.5):
        assert lag >= 1
        self.extractor = feature_extractor
        self.lag = lag
        self.w_loc = weight_location
        self.w_obj = weight_objects
        self._store: List[Dict[int, Dict[str, Any]]] = []

    def reset(self, batch_size: int):
        self._store = [dict() for _ in range(batch_size)]

    def record(self, env_idx: int, step: int, observation: str,
               admissible_actions: List[str], info: Dict[str, Any]):
        feats = self.extractor.extract_all(observation, admissible_actions, info)
        location = feats.get('location_change')
        objects = feats.get('visible_objects')
        self._store[env_idx][step] = {
            'location': location.value if location is not None else None,
            'objects': set(objects.value.get('objects', [])) if objects is not None else set(),
        }

    def anchor_step(self, current_step: int) -> Optional[int]:
        """当前步 T (1 起) 的 anchor 步号; 不可用时 None"""
        a = current_step - self.lag
        return a if a >= 1 else None

    def score(self, env_idx: int, anchor: int, parsed: Optional[Dict[str, Any]]) -> float:
        """回忆精度 ∈ [0,1]; 解析失败/无存档 → 0"""
        if parsed is None or anchor not in self._store[env_idx]:
            return 0.0
        truth = self._store[env_idx][anchor]
        total, got = 0.0, 0.0
        if 'location' in parsed:
            total += self.w_loc
            pred, act = parsed['location'], truth['location']
            if pred is None and act is None:
                got += self.w_loc
            elif pred is not None and act is not None \
                    and pred.lower().strip() == act.lower().strip():
                got += self.w_loc
        if 'objects' in parsed:
            total += self.w_obj
            pred_set, act_set = set(parsed['objects']), truth['objects']
            if not pred_set and not act_set:
                f1 = 1.0
            elif not pred_set or not act_set:
                f1 = 0.0
            else:
                tp = len(pred_set & act_set)
                p, r = tp / len(pred_set), tp / len(act_set)
                f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
            got += self.w_obj * f1
        return got / total if total > 0 else 0.0
