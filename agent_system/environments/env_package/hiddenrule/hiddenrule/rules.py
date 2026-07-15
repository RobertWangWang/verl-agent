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
HiddenRule-Gym 规则族 (docs/hiddenrule_gym_design.md §1.2)

四个规则族,分两类语义:
- 状态谓词类 (CONJ / XOR): 对当前机关状态即时求值,条件随状态变化可开可关;
- 事件序列类 (SEQ / COUNT): 对操作事件流求值,一旦满足即锁存 (latched)。

训练/探针分族 (§5.4 防泄漏) 依赖 family 字段。
"""

import random
from dataclasses import dataclass
from typing import List, Tuple

RULE_FAMILIES = ("conj", "seq", "xor", "count")


@dataclass(frozen=True)
class Rule:
    family: str
    # conj: conditions = ((device_name, required_state), ...)
    conditions: Tuple[Tuple[str, int], ...] = ()
    # xor: devices = (lever names...), parity = 0/1 (奇偶性要求)
    devices: Tuple[str, ...] = ()
    parity: int = 1
    # seq: sequence = (device names in required order)
    sequence: Tuple[str, ...] = ()
    # count: device + 需要的操作次数 n
    device: str = ""
    n: int = 0

    def describe(self, state_word) -> str:
        """规则的自然语言描述 (便签文本)。state_word(device_name, state)->str 由 world 提供。"""
        if self.family == "conj":
            parts = [f"{name} is {state_word(name, req)}" for name, req in self.conditions]
            return "The vault opens when " + " and ".join(parts) + "."
        if self.family == "xor":
            names = ", ".join(self.devices)
            odd = "an odd" if self.parity == 1 else "an even"
            return f"The vault opens when {odd} number of these levers are up: {names}."
        if self.family == "seq":
            steps = ", then ".join(self.sequence)
            return f"The vault unlocks after you operate, in this exact order: {steps}."
        if self.family == "count":
            return f"The vault unlocks after {self.device} has been operated {self.n} times."
        raise ValueError(f"unknown family {self.family}")


@dataclass(frozen=True)
class RuleProgress:
    """事件序列类规则的追踪状态 (进入 CoreState,必须可哈希)"""
    seq_progress: int = 0
    op_count: int = 0
    latched: bool = False


def update_on_op(rule: Rule, progress: RuleProgress, device_name: str) -> RuleProgress:
    """机关被操作 (toggle/set/press) 时推进事件类规则;状态类规则不受影响。"""
    if progress.latched:
        return progress
    if rule.family == "seq":
        if device_name == rule.sequence[progress.seq_progress]:
            nxt = progress.seq_progress + 1
            if nxt == len(rule.sequence):
                return RuleProgress(seq_progress=nxt, latched=True)
            return RuleProgress(seq_progress=nxt)
        # 操作错误: 进度重置 (若恰好是序列首元素则从 1 重新起步)
        restart = 1 if device_name == rule.sequence[0] else 0
        if restart == 1 and len(rule.sequence) == 1:
            return RuleProgress(seq_progress=1, latched=True)
        return RuleProgress(seq_progress=restart)
    if rule.family == "count":
        if device_name == rule.device:
            cnt = progress.op_count + 1
            if cnt >= rule.n:
                return RuleProgress(op_count=min(cnt, rule.n), latched=True)
            return RuleProgress(op_count=cnt)
        return progress
    return progress


def condition_active(rule: Rule, progress: RuleProgress, device_state_of) -> bool:
    """当前时刻规则条件是否满足。device_state_of(name)->int 读取机关状态。"""
    if rule.family == "conj":
        return all(device_state_of(name) == req for name, req in rule.conditions)
    if rule.family == "xor":
        ups = sum(1 for name in rule.devices if device_state_of(name) == 1)
        return ups % 2 == rule.parity
    return progress.latched


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------

def sample_rule(family: str, stateful_devices: List, button_devices: List,
                arity: int, rng: random.Random) -> Rule:
    """
    采样一条规则,保证初始状态 (全部机关归零) 下条件不成立 (episode 不平凡)。

    stateful_devices/button_devices: world.Device 列表 (lever/dial vs button)。
    """
    if family == "conj":
        chosen = rng.sample(stateful_devices, arity)
        conditions = []
        for i, dev in enumerate(chosen):
            req = rng.randrange(1, dev.n_states) if i == 0 else rng.randrange(0, dev.n_states)
            conditions.append((dev.name, req))
        # 首个条件强制非零 → 初始 (全零) 必不满足
        return Rule(family="conj", conditions=tuple(conditions))

    if family == "xor":
        levers = [d for d in stateful_devices if d.kind == "lever"]
        chosen = rng.sample(levers, min(arity, len(levers)))
        # 初始全 down → up 数为 0 (偶) → 要求奇数则初始不满足
        return Rule(family="xor", devices=tuple(d.name for d in chosen), parity=1)

    if family == "seq":
        pool = stateful_devices + button_devices
        chosen = rng.sample(pool, min(max(arity, 2), len(pool)))
        return Rule(family="seq", sequence=tuple(d.name for d in chosen))

    if family == "count":
        pool = button_devices if button_devices else stateful_devices
        dev = rng.choice(pool)
        return Rule(family="count", device=dev.name, n=rng.randrange(2, 4))

    raise ValueError(f"unknown family {family}")


def sample_fake_rule(true_rule: Rule, stateful_devices: List, button_devices: List,
                     arity: int, rng: random.Random, state_word, max_tries: int = 20) -> str:
    """采样一条与真规则文本不同的假规则描述 (干扰便签)。"""
    for _ in range(max_tries):
        family = rng.choice(RULE_FAMILIES)
        try:
            fake = sample_rule(family, stateful_devices, button_devices, arity, rng)
        except ValueError:
            continue
        text = fake.describe(state_word)
        if text != true_rule.describe(state_word):
            return text
    return "The vault opens at midnight."  # 兜底假线索
