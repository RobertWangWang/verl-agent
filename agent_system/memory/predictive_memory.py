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
预测充分性记忆模块 (Predictive Sufficiency Memory)

根据研究计划 §4.1:
L_pred(m_t) = −E[ log P_φ(o_{t+1:t+k} 的可验证特征 | m_t, a_t) ]

核心思想:
1. 记忆摘要 m_t 应该是未来观测的充分统计量
2. 预测准确率作为记忆质量的稠密奖励信号
3. 支持多步预测 (horizon k)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .base import BaseMemory


@dataclass
class PredictionResult:
    """
    预测结果数据结构
    """
    predicted_features: Dict[str, Any]  # 预测的特征
    confidence: float  # 预测置信度
    horizon: int  # 预测的步长


@dataclass
class PredictionReward:
    """
    预测奖励数据结构
    """
    turn: int  # 当前回合
    prediction_accuracy: float  # 预测准确率 (0-1)
    shaped_reward: float  # 整形后的奖励 (potential-based)
    component_rewards: Dict[str, float]  # 各特征维度的奖励


class PredictiveMemory(BaseMemory):
    """
    预测充分性记忆模块

    扩展自 SimpleMemory，增加预测能力:
    1. 存储历史交互 (同 SimpleMemory)
    2. 基于当前记忆预测未来特征
    3. 计算预测准确率作为奖励信号
    """

    REWARD_MODES = ('potential', 'raw', 'delta_clip')

    def __init__(
        self,
        prediction_horizon: int = 3,
        lambda_pred: float = 0.1,
        use_potential_shaping: bool = True,
        reward_mode: str = None
    ):
        """
        Args:
            prediction_horizon: 预测未来多少步的特征 (默认3步)
            lambda_pred: 预测奖励的权重系数
            use_potential_shaping: 是否使用 potential-based shaping (兼容参数;
                reward_mode 显式给出时以 reward_mode 为准)
            reward_mode: potential (Φ_t−Φ_{t−1}) | raw (Φ_t) |
                delta_clip (clip(Φ_t−Φ_{t−1}, 0, 1), R41 进步奖励:
                掌握后 Δ→0 且负向噪声被截断 → 组内方差按构造归零)
        """
        if reward_mode is None:
            reward_mode = 'potential' if use_potential_shaping else 'raw'
        assert reward_mode in self.REWARD_MODES, f"unknown reward_mode: {reward_mode}"
        self.prediction_horizon = prediction_horizon
        self.lambda_pred = lambda_pred
        self.use_potential_shaping = reward_mode == 'potential'
        self.reward_mode = reward_mode

        # 历史数据存储 (同 SimpleMemory)
        self._data = None
        self.keys = None
        self.batch_size = 0

        # 预测相关
        self.predictions: List[Dict[int, PredictionResult]] = []  # [env_idx][turn] -> PredictionResult
        self.prediction_rewards: List[Dict[int, PredictionReward]] = []  # [turn][env_idx] -> PredictionReward

        # 用于 potential-based shaping 的上一步预测准确率
        self.prev_prediction_accuracy: List[float] = []

    def __len__(self):
        return len(self._data) if self._data else 0

    def __getitem__(self, idx):
        return self._data[idx] if self._data else []

    def reset(self, batch_size: int):
        """重置记忆"""
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

        # 重置预测相关
        self.predictions = [{} for _ in range(batch_size)]
        self.prediction_rewards = []
        self.prev_prediction_accuracy = [0.0] * batch_size

    def store(self, record: Dict[str, List[Any]]):
        """
        存储新的交互记录 (同 SimpleMemory)

        Args:
            record: Dict[str, List[Any]]
                例如: {'text_obs': [...], 'action': [...]}
        """
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def fetch(
        self,
        history_length: int,
        obs_key: str = "text_obs",
        action_key: str = "action",
    ) -> Tuple[List[str], List[int]]:
        """
        获取历史记忆上下文 (同 SimpleMemory)

        Returns:
            memory_contexts: List[str] - 每个环境的记忆文本
            valid_lengths: List[int] - 有效历史长度
        """
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec[action_key]
                obs = rec[obs_key]
                lines.append(
                    f"[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"
                )

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths

    def record_prediction(
        self,
        env_idx: int,
        turn: int,
        predicted_features: Dict[str, Any],
        confidence: float = 1.0,
    ) -> PredictionResult:
        """
        记录预测头产出的预测，等待未来观测到来后验证。

        预测头是同骨干 LLM 的受限问答: response 中的 <predict> 块，
        由 verifiable_features.parse_predict_block + prediction_to_features
        解析成 {feature_type: VerifiableFeature} (规则解析，不经 LLM judge)。

        对应研究计划 §4.1:
        L_pred(m_t) = −E[ log P_φ(o_{t+1:t+k} 的可验证特征 | m_t, a_t) ]

        Args:
            env_idx: 环境索引
            turn: 做出预测的回合
            predicted_features: {feature_type: VerifiableFeature}，
                                来自 prediction_to_features 的解析结果
            confidence: 预测置信度 (预留，后续可用输出概率校准)

        Returns:
            PredictionResult: 记录下的预测
        """
        result = PredictionResult(
            predicted_features=predicted_features,
            confidence=confidence,
            horizon=self.prediction_horizon
        )
        self.predictions[env_idx][turn] = result
        return result

    def verify_prediction(
        self,
        env_idx: int,
        turn: int,
        feature_extractor,
        actual_obs: str,
        actual_actions: List[str],
        actual_info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, bool]]:
        """
        验证预测是否正确

        Args:
            env_idx: 环境索引
            turn: 要验证的回合
            feature_extractor: 特征提取器
            actual_obs: 实际观测
            actual_actions: 实际可用动作
            actual_info: 实际环境信息

        Returns:
            accuracy: 加权预测准确率 (0-1)，权重来自 feature_extractor
                      (例如 task_progress 默认权重 0，恒对的平凡预测不计入)
            component_results: 各特征的验证结果
        """
        # 获取之前的预测
        if turn not in self.predictions[env_idx]:
            # 没有预测，返回0准确率
            return 0.0, {}

        prediction = self.predictions[env_idx][turn]
        predicted_features = prediction.predicted_features

        # 提取实际特征
        actual_features = feature_extractor.extract_all(
            actual_obs, actual_actions, actual_info
        )

        # 验证预测 (逐特征结果) 与加权准确率
        verification_results = feature_extractor.verify_all(
            predicted_features, actual_features
        )
        accuracy = feature_extractor.compute_reward(
            predicted_features, actual_features
        )

        return accuracy, verification_results

    def compute_prediction_reward(
        self,
        env_idx: int,
        turn: int,
        current_accuracy: float,
    ) -> PredictionReward:
        """
        计算预测奖励 (potential-based shaping)

        根据 research proposal §4.3:
        r_pred(t) = Φ(s_t) - γΦ(s_{t-1})
        其中 Φ(s_t) 是预测准确率

        Args:
            env_idx: 环境索引
            turn: 当前回合
            current_accuracy: 当前预测准确率

        Returns:
            PredictionReward: 预测奖励
        """
        if self.reward_mode == 'potential':
            # Potential-based shaping
            prev_accuracy = self.prev_prediction_accuracy[env_idx]
            shaped_reward = current_accuracy - prev_accuracy  # Φ(s_t) - Φ(s_{t-1})
        elif self.reward_mode == 'delta_clip':
            # R41 进步奖励: 只奖励准确率的提升, 回落不惩罚 (clip 下界 0)。
            # 掌握后 Δ≈0、噪声负向被截 → 全败组内方差按构造归零, std 归一化无弹药可放大。
            prev_accuracy = self.prev_prediction_accuracy[env_idx]
            shaped_reward = min(1.0, max(0.0, current_accuracy - prev_accuracy))
        else:
            # raw: 直接使用准确率作为奖励
            shaped_reward = current_accuracy

        # 加权
        weighted_reward = self.lambda_pred * shaped_reward

        reward = PredictionReward(
            turn=turn,
            prediction_accuracy=current_accuracy,
            shaped_reward=weighted_reward,
            component_rewards={}  # 可选: 各特征维度的奖励
        )

        # 更新上一步准确率
        self.prev_prediction_accuracy[env_idx] = current_accuracy

        return reward

    def get_prediction_rewards(self) -> List[Dict[int, PredictionReward]]:
        """
        获取所有预测奖励

        Returns:
            List[Dict[int, PredictionReward]]: [turn][env_idx] -> reward
        """
        return self.prediction_rewards

    def get_summary_for_prompt(
        self,
        env_idx: int,
        history_length: int = 5
    ) -> str:
        """
        获取记忆摘要，用于插入到 prompt 中

        这是"预测充分性"的核心: 记忆摘要应该是
        对未来观测预测有用的信息

        Args:
            env_idx: 环境索引
            history_length: 要总结的历史长度

        Returns:
            str: 记忆摘要文本
        """
        if len(self._data) <= env_idx or len(self._data[env_idx]) == 0:
            return "No history yet."

        recent = self._data[env_idx][-history_length:]

        # 简化版本: 直接拼接历史
        # 后续可以实现更智能的摘要:
        # 1. 提取关键物体
        # 2. 总结位置变化
        # 3. 标注重要事件
        summary_parts = []
        for i, rec in enumerate(recent):
            obs = rec.get('text_obs', '')
            action = rec.get('action', '')
            summary_parts.append(f"Step: {action} → Obs: {obs[:50]}...")

        return "\n".join(summary_parts)


class HybridMemory(BaseMemory):
    """
    混合记忆模块

    结合 SimpleMemory 和 PredictiveMemory:
    - 使用 SimpleMemory 的存储/获取逻辑
    - 添加 PredictiveMemory 的预测能力
    """

    def __init__(
        self,
        history_length: int = 2,
        prediction_horizon: int = 3,
        lambda_pred: float = 0.1,
        enable_prediction: bool = True,
        reward_mode: str = None
    ):
        """
        Args:
            history_length: 历史记忆长度
            prediction_horizon: 预测步长
            lambda_pred: 预测奖励权重
            enable_prediction: 是否启用预测功能
            reward_mode: 见 PredictiveMemory.reward_mode (None = potential)
        """
        self.history_length = history_length
        self.enable_prediction = enable_prediction

        # 使用 PredictiveMemory 作为底层实现
        self.memory = PredictiveMemory(
            prediction_horizon=prediction_horizon,
            lambda_pred=lambda_pred,
            reward_mode=reward_mode
        )

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        return self.memory[idx]

    def reset(self, batch_size: int):
        self.memory.reset(batch_size)

    def store(self, record: Dict[str, List[Any]]):
        self.memory.store(record)

    def fetch(
        self,
        history_length: int = None,
        obs_key: str = "text_obs",
        action_key: str = "action",
    ) -> Tuple[List[str], List[int]]:
        """
        获取历史记忆上下文
        """
        if history_length is None:
            history_length = self.history_length
        return self.memory.fetch(history_length, obs_key, action_key)

    # 预测相关方法
    def record_prediction(self, *args, **kwargs):
        """委托给 PredictiveMemory"""
        if not self.enable_prediction:
            return None
        return self.memory.record_prediction(*args, **kwargs)

    def verify_prediction(self, *args, **kwargs):
        """委托给 PredictiveMemory"""
        if not self.enable_prediction:
            return 0.0, {}
        return self.memory.verify_prediction(*args, **kwargs)

    def compute_prediction_reward(self, *args, **kwargs):
        """委托给 PredictiveMemory"""
        if not self.enable_prediction:
            return None
        return self.memory.compute_prediction_reward(*args, **kwargs)

    def get_prediction_rewards(self):
        """委托给 PredictiveMemory"""
        if not self.enable_prediction:
            return []
        return self.memory.get_prediction_rewards()
