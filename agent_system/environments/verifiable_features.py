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
可验证特征提取器 (Verifiable Feature Extractor)

根据研究计划，我们需要从环境观测中提取"可自动验证的离散特征"，
这些特征可以：
1. 由规则程序自动判断真假
2. 不需要 LLM judge
3. 可以作为预测充分性的目标

核心思想：预测充分性 ⟺ 记忆摘要能否支撑对未来观测的预测
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class VerifiableFeature:
    """
    可验证特征的数据结构
    """
    feature_type: str  # 特征类型 (object_seen, location_change, action_available, etc.)
    value: Any  # 特征值
    confidence: float = 1.0  # 预测置信度 (可选)
    metadata: Dict[str, Any] = None  # 额外元数据


class BaseFeatureExtractor(ABC):
    """
    特征提取器基类

    子类必须定义 feature_type 类属性，它是特征的唯一标识:
    - extract() 产出的 VerifiableFeature.feature_type 必须等于它
    - CompositeFeatureExtractor 用它做 提取器 <-> 特征 的匹配和加权
    """

    feature_type: str = None  # 子类必须覆盖

    @abstractmethod
    def extract(self, observation: str, admissible_actions: List[str], info: Dict[str, Any]) -> VerifiableFeature:
        """
        从当前观测中提取可验证特征

        Args:
            observation: 当前文本观测
            admissible_actions: 当前可用动作列表
            info: 环境返回的额外信息

        Returns:
            VerifiableFeature: 提取的特征
        """
        pass

    @abstractmethod
    def verify(self, predicted: VerifiableFeature, actual: VerifiableFeature) -> bool:
        """
        验证预测特征是否与实际特征匹配

        Args:
            predicted: 预测的特征
            actual: 实际的特征

        Returns:
            bool: 预测是否正确
        """
        pass


class ALFWorldObjectSeenFeature(BaseFeatureExtractor):
    """
    ALFWorld 物体可见性特征

    预测目标：下一个观测中是否会出现特定物体
    可验证性：通过文本匹配自动判断
    """

    feature_type = 'object_seen'

    def __init__(self, target_objects: Set[str]):
        """
        Args:
            target_objects: 需要追踪的目标物体集合
                           例如: {'ladle', 'fridge', 'knife', 'apple'}
        """
        self.target_objects = target_objects

    def extract(self, observation: str, admissible_actions: List[str], info: Dict[str, Any]) -> VerifiableFeature:
        """
        从观测中提取哪些物体被看到

        示例观测:
        "You open the cabinet 1. The cabinet 1 is open. In it, you see nothing."
        "You arrive at fridge 1. The fridge 1 is closed."

        返回: feature_type='object_seen', value={'seen': [], 'not_seen': ['ladle']}
        """
        obs_lower = observation.lower()
        seen_objects = set()
        mentioned_objects = set()

        # 匹配 "you see X" 或 "in it you see X" (更鲁棒的匹配)
        for obj in self.target_objects:
            obj_lower = obj.lower()
            # 使用正则表达式进行更灵活的匹配
            # 匹配: "see ... object" (中间可以有任意内容，如 "see a ladle and a knife")
            patterns = [
                rf'see.*{re.escape(obj_lower)}',  # "see ... object" (最宽松)
                rf'seeing.*{re.escape(obj_lower)}',
            ]
            if any(re.search(pattern, obs_lower) for pattern in patterns):
                seen_objects.add(obj)
            # 检查物体是否被提及（更宽松的匹配）
            if obj_lower in obs_lower.split():
                mentioned_objects.add(obj)

        return VerifiableFeature(
            feature_type=self.feature_type,
            value={
                'seen': list(seen_objects),
                'mentioned': list(mentioned_objects),
                'not_seen': list(self.target_objects - seen_objects)
            },
            metadata={'observation_length': len(observation)}
        )

    def verify(self, predicted: VerifiableFeature, actual: VerifiableFeature) -> bool:
        """
        验证预测的物体可见性是否正确

        支持两种预测形式:
        1. {'visible': bool} —— 来自 <predict> 块的受限预测 (target_visible: yes/no)，
           判断"是否有目标物体可见"这一布尔命题是否正确
        2. {'seen': [...]} —— 具体物体列表 (legacy)，判断预测的物体是否都实际可见
        """
        if predicted.feature_type != self.feature_type or actual.feature_type != self.feature_type:
            raise ValueError("Feature type mismatch")

        pred_val = predicted.value or {}
        actual_seen = set(actual.value.get('seen', []))

        if 'visible' in pred_val:
            return bool(actual_seen) == bool(pred_val['visible'])

        pred_seen = set(pred_val.get('seen', []))
        return pred_seen.issubset(actual_seen)


class ALFWorldLocationChangeFeature(BaseFeatureExtractor):
    """
    ALFWorld 位置变化特征

    预测目标：下一个观测中 agent 是否到达特定位置
    可验证性：通过 "You arrive at X" 或 "You are at X" 模式匹配
    """

    feature_type = 'location_change'

    # ALFWorld 常见位置类型
    LOCATION_PATTERNS = {
        'cabinet': r'cabinet\s*\d+',
        'fridge': r'fridge\s*\d+',
        'drawer': r'drawer\s*\d+',
        'countertop': r'countertop\s*\d+',
        'diningtable': r'diningtable\s*\d+',
        'sinkbasin': r'sinkbasin\s*\d+',
        'stoveburner': r'stoveburner\s*\d+',
        'microwave': r'microwave\s*\d+',
        'toaster': r'toaster\s*\d+',
        'garbagecan': r'garbagecan\s*\d+',
        'coffeemachine': r'coffeemachine\s*\d+',
    }

    def extract(self, observation: str, admissible_actions: List[str], info: Dict[str, Any]) -> VerifiableFeature:
        """
        从观测中提取当前位置信息

        示例观测:
        "You arrive at cabinet 1."
        "You go to diningtable 1."

        返回: feature_type='location_change', value='cabinet 1'
        """
        obs_lower = observation.lower()
        current_location = None

        # 匹配 "You arrive at X" 或 "You go to X"
        for loc_type, pattern in self.LOCATION_PATTERNS.items():
            matches = re.findall(pattern, obs_lower, re.IGNORECASE)
            if matches:
                # 提取位置信息
                for match in matches:
                    if f'arrive at {loc_type}' in obs_lower or f'go to {loc_type}' in obs_lower:
                        current_location = f"{loc_type} {match.split()[-1]}"
                        break
                if current_location:
                    break

        # 也可以从 admissible_actions 中推断位置
        # 例如: "go to cabinet 1" 表示当前不在 cabinet 1

        return VerifiableFeature(
            feature_type=self.feature_type,
            value=current_location,
            metadata={
                'admissible_action_count': len(admissible_actions),
                'has_goto': any('go to' in action.lower() for action in admissible_actions)
            }
        )

    def verify(self, predicted: VerifiableFeature, actual: VerifiableFeature) -> bool:
        """
        验证预测的位置变化是否正确

        预测: "go to fridge 1" → 下一个位置是 "fridge 1"
        实际: "You arrive at fridge 1"
        结果: True
        """
        if predicted.feature_type != self.feature_type or actual.feature_type != self.feature_type:
            raise ValueError("Feature type mismatch")

        pred_loc = predicted.value
        actual_loc = actual.value

        if pred_loc is None and actual_loc is None:
            return True
        if pred_loc is None or actual_loc is None:
            return False

        # 标准化位置字符串后比较
        return pred_loc.lower().strip() == actual_loc.lower().strip()


class ALFWorldActionAvailabilityFeature(BaseFeatureExtractor):
    """
    ALFWorld 动作可用性特征

    预测目标：下一个观测中特定动作是否可用
    可验证性：检查 admissible_actions 列表
    """

    feature_type = 'action_available'

    # 需要追踪的关键动作模式
    ACTION_PATTERNS = {
        'pick': r'pick\s+up\s+\w+',
        'put': r'put\s+\w+\s+in\s+\w+|move\s+\w+\s+to\s+\w+',
        'open': r'open\s+\w+',
        'close': r'close\s+\w+',
        'toggle': r'toggle\s+\w+',
        'heat': r'heat\s+\w+',
        'clean': r'clean\s+\w+',
        'cool': r'cool\s+\w+',
        'slice': r'slice\s+\w+',
        'examine': r'examine\s+\w+',
    }

    def extract(self, observation: str, admissible_actions: List[str], info: Dict[str, Any]) -> VerifiableFeature:
        """
        从可用动作列表中提取动作可用性信息

        返回: feature_type='action_available', value={'pick': True, 'put': False, ...}
        """
        available_patterns = set()
        actions_str = ' '.join(admissible_actions).lower()

        for action_type, pattern in self.ACTION_PATTERNS.items():
            if re.search(pattern, actions_str):
                available_patterns.add(action_type)

        return VerifiableFeature(
            feature_type=self.feature_type,
            value=list(available_patterns),
            metadata={
                'total_admissible_actions': len(admissible_actions),
                'actions_sample': admissible_actions[:5] if admissible_actions else []
            }
        )

    def verify(self, predicted: VerifiableFeature, actual: VerifiableFeature) -> bool:
        """
        验证预测的动作可用性是否正确
        """
        if predicted.feature_type != self.feature_type or actual.feature_type != self.feature_type:
            raise ValueError("Feature type mismatch")

        pred_actions = set(predicted.value)
        actual_actions = set(actual.value)

        # 计算预测的动作是否在实际中可用
        # 这里可以有不同的判断策略：
        # 1. 精确匹配：所有预测动作都在实际中
        # 2. 部分匹配：至少有一个预测动作在实际中
        # 3. 覆盖率：计算预测动作在实际中的比例

        # 使用精确匹配作为基准
        return pred_actions.issubset(actual_actions)


class ALFWorldTaskProgressFeature(BaseFeatureExtractor):
    """
    ALFWorld 任务进度特征

    预测目标：任务是否完成或接近完成
    可验证性：检查 info['won'] 或 info['goal_condition_success_rate']

    注意: 绝大多数 step 预测 "未完成" 都是平凡正确的，会虚增准确率。
    P1 阶段默认权重为 0 (见 create_alfworld_feature_extractor)，仅记日志。
    """

    feature_type = 'task_progress'

    def extract(self, observation: str, admissible_actions: List[str], info: Dict[str, Any]) -> VerifiableFeature:
        """
        从 info 中提取任务进度信息

        返回: feature_type='task_progress', value={'won': bool, 'success_rate': float}
        """
        won = info.get('won', False)
        goal_success_rate = info.get('goal_condition_success_rate', 0.0)

        return VerifiableFeature(
            feature_type=self.feature_type,
            value={
                'won': bool(won),
                'success_rate': float(goal_success_rate)
            },
            metadata={
                'reward': info.get('reward', 0.0)
            }
        )

    def verify(self, predicted: VerifiableFeature, actual: VerifiableFeature) -> bool:
        """
        验证预测的任务进度是否正确
        """
        if predicted.feature_type != self.feature_type or actual.feature_type != self.feature_type:
            raise ValueError("Feature type mismatch")

        pred_won = predicted.value.get('won', False)
        actual_won = actual.value.get('won', False)

        # 对于任务完成，使用精确匹配
        return pred_won == actual_won


class CompositeFeatureExtractor:
    """
    组合特征提取器

    将多个特征提取器组合在一起，提供统一接口
    """

    def __init__(self, extractors: List[Tuple[BaseFeatureExtractor, float]]):
        """
        Args:
            extractors: (提取器, 权重) 列表，权重用于计算综合奖励
        """
        self.extractors = extractors

    def extract_all(
        self,
        observation: str,
        admissible_actions: List[str],
        info: Dict[str, Any]
    ) -> Dict[str, VerifiableFeature]:
        """
        提取所有特征

        Returns:
            Dict[str, VerifiableFeature]: {feature_type: feature}
        """
        features = {}
        for extractor, _ in self.extractors:
            feature = extractor.extract(observation, admissible_actions, info)
            features[feature.feature_type] = feature
        return features

    def verify_all(
        self,
        predicted_features: Dict[str, VerifiableFeature],
        actual_features: Dict[str, VerifiableFeature]
    ) -> Dict[str, bool]:
        """
        验证所有预测特征。只验证预测和实际中都存在的特征
        (预测缺某个特征时该特征不计入，权重在 compute_reward 中相应归一化)。

        Returns:
            Dict[str, bool]: {feature_type: is_correct}
        """
        results = {}
        extractor_map = {extractor.feature_type: extractor for extractor, _ in self.extractors}

        for feature_type, pred_feature in predicted_features.items():
            if feature_type in actual_features and feature_type in extractor_map:
                results[feature_type] = extractor_map[feature_type].verify(
                    pred_feature, actual_features[feature_type]
                )
        return results

    def compute_reward(
        self,
        predicted_features: Dict[str, VerifiableFeature],
        actual_features: Dict[str, VerifiableFeature]
    ) -> float:
        """
        基于预测准确率计算加权综合分数

        只对实际参与验证的特征做加权归一化;
        若被验证的特征权重之和为 0 (例如只预测了权重为 0 的特征)，返回 0.0。

        Returns:
            float: 综合分数 (0-1)
        """
        verification_results = self.verify_all(predicted_features, actual_features)

        total_weight = 0.0
        weighted_score = 0.0

        for extractor, weight in self.extractors:
            if extractor.feature_type in verification_results:
                weighted_score += weight * float(verification_results[extractor.feature_type])
                total_weight += weight

        if total_weight > 0:
            return weighted_score / total_weight
        return 0.0


# ---------------------------------------------------------------------------
# ALFWorld 任务目标物体解析
# ---------------------------------------------------------------------------

# ALFWorld (ALFRED) 标准物体/容器词表，用于从任务描述中解析任务相关物体。
# prompt 中 target_visible 的语义是 "任务描述中提到的物体是否可见"，
# 验证时目标集合必须与之对齐 (docs/ps_grpo_integration_design.md §2B)。
ALFWORLD_OBJECT_VOCAB = {
    'alarmclock', 'apple', 'armchair', 'baseballbat', 'basketball', 'bathtubbasin',
    'bed', 'book', 'boots', 'bowl', 'box', 'bread', 'butterknife', 'cabinet',
    'candle', 'cart', 'cd', 'cellphone', 'cloth', 'coffeemachine', 'coffeetable',
    'countertop', 'creditcard', 'cup', 'desk', 'desklamp', 'diningtable',
    'dishsponge', 'drawer', 'dresser', 'egg', 'floorlamp', 'fork', 'fridge',
    'garbagecan', 'glassbottle', 'handtowel', 'handtowelholder', 'houseplant',
    'kettle', 'keychain', 'knife', 'ladle', 'laptop', 'lettuce', 'microwave',
    'mug', 'newspaper', 'ottoman', 'pan', 'papertowelroll', 'pen', 'pencil',
    'peppershaker', 'pillow', 'plate', 'plunger', 'pot', 'potato',
    'remotecontrol', 'safe', 'saltshaker', 'shelf', 'sidetable', 'sinkbasin',
    'soapbar', 'soapbottle', 'sofa', 'spatula', 'spoon', 'spraybottle',
    'statue', 'stoveburner', 'tissuebox', 'toilet', 'toiletpaper',
    'toiletpaperhanger', 'tomato', 'towel', 'towelholder', 'tvstand', 'vase',
    'watch', 'wateringcan', 'winebottle',
}


def task_target_objects(task_description: str) -> Set[str]:
    """
    从 ALFWorld 任务描述中解析任务相关的物体集合。

    例如 "put a clean ladle in diningtable" → {'ladle', 'diningtable'}
    解析不出任何物体时返回空集 (调用方回退到默认物体集)。
    """
    if not task_description:
        return set()
    tokens = re.findall(r'[a-z]+', task_description.lower())
    return {t for t in tokens if t in ALFWORLD_OBJECT_VOCAB}


# ---------------------------------------------------------------------------
# <predict> 块解析 (规则程序，不经过 LLM judge)
# ---------------------------------------------------------------------------

PREDICT_BLOCK_RE = re.compile(r'<predict>(.*?)</predict>', re.DOTALL | re.IGNORECASE)

_YES_VALUES = {'yes', 'true', 'y', '1'}
_NO_VALUES = {'no', 'false', 'n', '0'}
_NONE_VALUES = {'none', 'no change', 'n/a', 'na', 'same', 'unchanged', ''}


def parse_predict_block(text: str) -> Optional[Dict[str, Any]]:
    """
    从 LLM response 文本中解析 <predict> 块。

    期望格式 (见 ALFWORLD_TEMPLATE_PS):
        <predict>next_location: cabinet 2; target_visible: yes; task_done: no</predict>

    解析规则:
    - 缺少 <predict> 块，或块内没有任何可识别字段 → 返回 None (由调用方记为解析失败)
    - next_location: "none"/"no change" 等 → None (表示位置不变/无位置)
    - target_visible / task_done: yes 系 → True，no 系 → False，
      其他无法判定的值 → 该字段省略 (不参与验证，不猜)

    Returns:
        Dict 或 None。可能的键: 'next_location', 'target_visible', 'task_done'
    """
    if not text:
        return None
    match = PREDICT_BLOCK_RE.search(text)
    if match is None:
        return None

    parsed: Dict[str, Any] = {}
    for part in match.group(1).split(';'):
        if ':' not in part:
            continue
        key, value = part.split(':', 1)
        key = key.strip().lower()
        value = value.strip().lower().strip("'\"")

        if key == 'next_location':
            parsed['next_location'] = None if value in _NONE_VALUES else value
        elif key == 'target_visible':
            if value in _YES_VALUES:
                parsed['target_visible'] = True
            elif value in _NO_VALUES:
                parsed['target_visible'] = False
        elif key == 'task_done':
            if value in _YES_VALUES:
                parsed['task_done'] = True
            elif value in _NO_VALUES:
                parsed['task_done'] = False

    return parsed if parsed else None


def prediction_to_features(parsed: Dict[str, Any]) -> Dict[str, VerifiableFeature]:
    """
    把 parse_predict_block 的结果转成可与 extract_all 输出直接对比验证的
    {feature_type: VerifiableFeature} 字典。省略的字段不生成特征 (不参与验证)。
    """
    features: Dict[str, VerifiableFeature] = {}
    if 'next_location' in parsed:
        features['location_change'] = VerifiableFeature(
            feature_type='location_change', value=parsed['next_location']
        )
    if 'target_visible' in parsed:
        features['object_seen'] = VerifiableFeature(
            feature_type='object_seen', value={'visible': parsed['target_visible']}
        )
    if 'task_done' in parsed:
        features['task_progress'] = VerifiableFeature(
            feature_type='task_progress', value={'won': parsed['task_done']}
        )
    return features


def create_alfworld_feature_extractor(
    object_types: Set[str] = None,
    feature_weights: Dict[str, float] = None
) -> CompositeFeatureExtractor:
    """
    创建 ALFWorld 的组合特征提取器

    Args:
        object_types: 需要追踪的物体类型
        feature_weights: 各特征的权重

    Returns:
        CompositeFeatureExtractor: 组合特征提取器
    """
    if object_types is None:
        # 默认追踪的物体类型
        object_types = {
            'ladle', 'knife', 'fork', 'spoon',  # 餐具
            'apple', 'banana', 'bread', 'egg',  # 食物
            'plate', 'bowl', 'cup', 'mug',  # 容器
            'fridge', 'cabinet', 'drawer',  # 家具
        }

    if feature_weights is None:
        # task_progress 默认权重 0: "预测任务未完成"几乎恒对，会虚增准确率
        # (设计文档 docs/ps_grpo_integration_design.md §3)，仅保留用于日志
        feature_weights = {
            'object_seen': 0.4,      # 物体可见性
            'location_change': 0.4,  # 位置变化
            'action_available': 0.2, # 动作可用性
            'task_progress': 0.0,    # 任务进度 (仅日志)
        }

    extractors = [
        (ALFWorldObjectSeenFeature(object_types), feature_weights.get('object_seen', 0.4)),
        (ALFWorldLocationChangeFeature(), feature_weights.get('location_change', 0.4)),
        (ALFWorldActionAvailabilityFeature(), feature_weights.get('action_available', 0.2)),
        (ALFWorldTaskProgressFeature(), feature_weights.get('task_progress', 0.0)),
    ]

    return CompositeFeatureExtractor(extractors)
