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
HiddenRule-Gym 状态机与 gym 风格环境 (docs/hiddenrule_gym_design.md §1.3)

核心设计: transition/admissible 是**纯函数**,env (在线交互) 与 oracle (BFS 全知解算)
共用同一份转移语义,杜绝两者分叉。
"""

import random
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

from .render import render_observation
from .rules import RuleProgress, condition_active, update_on_op
from .world import HRGConfig, World, generate_world

VAULT_REWARD = 10.0


@dataclass(frozen=True)
class CoreState:
    room: int
    device_states: Tuple[int, ...]        # 与 world.stateful_devices 对齐
    progress: RuleProgress
    inventory: frozenset                  # 已拾取物品名
    unlocked_doors: frozenset             # 已用钥匙打开的门下标
    notes_read: frozenset                 # 已读便签名
    vault_open: bool


def initial_state(world: World) -> CoreState:
    return CoreState(
        room=world.start_room,
        device_states=tuple(0 for _ in world.stateful_devices),
        progress=RuleProgress(),
        inventory=frozenset(),
        unlocked_doors=frozenset(),
        notes_read=frozenset(),
        vault_open=False,
    )


def _door_passable(world: World, state: CoreState, door_idx: int) -> bool:
    door = world.doors[door_idx]
    return door.kind == "open" or door_idx in state.unlocked_doors


def _device_state_of(world: World, state: CoreState, name: str) -> int:
    return state.device_states[world.stateful_index[name]]


def vault_openable(world: World, state: CoreState) -> bool:
    return condition_active(world.rule, state.progress,
                            lambda name: _device_state_of(world, state, name))


def admissible_actions(world: World, state: CoreState) -> List[str]:
    actions: List[str] = []
    # 移动 (只列可通行的门; 锁着的门在观测里可见但不可走)
    for i, door in enumerate(world.doors):
        if door.connects(state.room) and _door_passable(world, state, i):
            actions.append(f"go to room {door.other(state.room) + 1}")
    # 机关操作
    for dev in world.devices:
        if dev.room != state.room:
            continue
        if dev.kind == "lever":
            actions.append(f"toggle {dev.name}")
        elif dev.kind == "dial":
            cur = _device_state_of(world, state, dev.name)
            actions.extend(f"set {dev.name} to {v}" for v in range(dev.n_states) if v != cur)
        else:
            actions.append(f"press {dev.name}")
    # 便签与物品
    actions.extend(f"read {note.name}" for note in world.notes if note.room == state.room)
    actions.extend(f"pick up {item.name}" for item in world.items
                   if item.room == state.room and item.name not in state.inventory)
    # 用钥匙开门
    for i, door in enumerate(world.doors):
        if (door.kind == "key" and door.connects(state.room)
                and i not in state.unlocked_doors and door.key_name in state.inventory):
            actions.append(f"use {door.key_name} on door to room {door.other(state.room) + 1}")
    # 保险库
    if state.room == world.vault_room and not state.vault_open:
        actions.append("open vault")
    actions.append("look")
    return actions


def transition(world: World, state: CoreState, action: str) -> Tuple[CoreState, str, bool]:
    """
    执行动作。返回 (新状态, 反馈文本, 动作是否有效)。
    无效动作 (不在 admissible 集) 一律原地不动,反馈 "Nothing happens."

    匹配大小写不敏感: 机关名含大写字母 (lever_E),而上游 projection 会把
    LLM 输出统一转小写 —— 精确匹配曾导致 manager 通路上所有动作被判无效
    (HRG-d 首跑 0/32 事故,见 research_logs)。按小写归一到唯一的规范动作。
    """
    action = action.strip()
    candidates = admissible_actions(world, state)
    if action not in candidates:
        lowered = {c.lower(): c for c in candidates}
        canonical = lowered.get(action.lower())
        if canonical is None:
            return state, "Nothing happens.", False
        action = canonical

    if action == "look":
        return state, "You look around.", True

    if action == "open vault":
        if vault_openable(world, state):
            return replace(state, vault_open=True), "The vault swings open!", True
        return state, "The vault does not budge.", True

    verb, _, rest = action.partition(" ")

    if action.startswith("go to room "):
        target = int(action.rsplit(" ", 1)[1]) - 1
        return replace(state, room=target), f"You move to room {target + 1}.", True

    if action.startswith("read "):
        note = next(n for n in world.notes if n.name == rest)
        new_state = replace(state, notes_read=state.notes_read | {note.name})
        return new_state, f"{note.name} reads: \"{note.text}\"", True

    if action.startswith("pick up "):
        item_name = action[len("pick up "):]
        new_state = replace(state, inventory=state.inventory | {item_name})
        return new_state, f"You pick up the {item_name}.", True

    if action.startswith("use "):
        # "use <key> on door to room <k>"
        key_name = action.split(" ")[1]
        target = int(action.rsplit(" ", 1)[1]) - 1
        for i, door in enumerate(world.doors):
            if (door.kind == "key" and door.connects(state.room)
                    and door.other(state.room) == target and door.key_name == key_name):
                new_state = replace(state, unlocked_doors=state.unlocked_doors | {i})
                return new_state, f"You unlock the door to room {target + 1}.", True
        return state, "Nothing happens.", False

    # 机关操作: toggle / set / press → 更新状态 + 推进事件类规则
    if verb == "toggle":
        dev_name = rest
        idx = world.stateful_index[dev_name]
        states = list(state.device_states)
        states[idx] = 1 - states[idx]
        new_state = replace(state, device_states=tuple(states),
                            progress=update_on_op(world.rule, state.progress, dev_name))
        word = world.state_word(dev_name, states[idx])
        return new_state, f"{dev_name} is now {word}.", True

    if verb == "set":
        # "set <dial> to <v>"
        parts = action.split(" ")
        dev_name, value = parts[1], int(parts[3])
        idx = world.stateful_index[dev_name]
        states = list(state.device_states)
        states[idx] = value
        new_state = replace(state, device_states=tuple(states),
                            progress=update_on_op(world.rule, state.progress, dev_name))
        return new_state, f"{dev_name} is now set to {value}.", True

    if verb == "press":
        dev_name = rest
        new_state = replace(state, progress=update_on_op(world.rule, state.progress, dev_name))
        return new_state, f"You press {dev_name}. Click.", True

    return state, "Nothing happens.", False


class HiddenRuleEnv:
    """gym 风格单环境。reset(seed) 完全确定性。"""

    # coverage_level < 1.0 时的 (mask, C) 缓存: 同 seed 同 config 的 world 完全确定,
    # 组环境语义下同组 worker 各自算一遍但结果一致; 缓存防同 worker 内 seed 复用重算
    _coverage_cache: Dict[int, Tuple[Tuple[str, ...], float]] = {}
    _COVERAGE_CACHE_MAX = 256

    def __init__(self, config: Optional[HRGConfig] = None):
        self.config = config or HRGConfig()
        self.world: Optional[World] = None
        self.state: Optional[CoreState] = None
        self.steps = 0
        self._oracle_steps: Optional[int] = None
        self._noise_rng: Optional[random.Random] = None
        self._phi_mask: Optional[Tuple[str, ...]] = None
        self._phi_coverage: Optional[float] = None
        self._evidence_revealed_at: Optional[int] = None
        self._decisive_mistake_at: Optional[int] = None

    def reset(self, seed: int) -> Tuple[str, Dict]:
        from .oracle import solve  # 延迟导入避免环
        self.world = generate_world(self.config, seed)
        self.state = initial_state(self.world)
        self.steps = 0
        # 观测噪声专用 rng: 与布局 rng 独立,同 seed 同动作序列 → 观测逐字节可复现
        self._noise_rng = random.Random(seed ^ 0x5EED11)
        # HRG-e 因果标注 (设计 §3): 证据揭示回合 / 决定性失误回合
        self._evidence_revealed_at = None
        self._decisive_mistake_at = None
        solution = solve(self.world)
        self._oracle_steps = len(solution) if solution is not None else -1
        # C-sweep (主图 1): coverage_level < 1.0 时把预测目标 Φ 裁剪到贪心校准的
        # 字段 mask; =1.0 走零成本路径 (不枚举状态空间, mask=None 表示不裁剪)
        if self.config.coverage_level < 1.0 - 1e-9:
            self._phi_mask, self._phi_coverage = self._calibrate_phi_mask(seed)
        else:
            self._phi_mask, self._phi_coverage = None, None
        obs = render_observation(self.world, self.state, "You enter the rooms.",
                                 self.config, rng=self._noise_rng)
        return obs, self._info(action_valid=True)

    def _calibrate_phi_mask(self, seed: int) -> Tuple[Tuple[str, ...], float]:
        from .coverage import calibrate_masks, coverage, enumerate_reachable, sweep_fields
        cached = HiddenRuleEnv._coverage_cache.get(seed)
        if cached is not None:
            return cached
        level = self.config.coverage_level
        states = enumerate_reachable(self.world)
        ladder = calibrate_masks(self.world, targets=(level,), states=states,
                                 fields=sweep_fields(self.world))
        mask = ladder[level]
        achieved = coverage(self.world, mask, states=states)
        result = (tuple(sorted(mask)), round(achieved, 4))
        if len(HiddenRuleEnv._coverage_cache) >= HiddenRuleEnv._COVERAGE_CACHE_MAX:
            HiddenRuleEnv._coverage_cache.clear()
        HiddenRuleEnv._coverage_cache[seed] = result
        return result

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        assert self.world is not None, "call reset() first"
        pre_openable = vault_openable(self.world, self.state)
        self.state, feedback, valid = transition(self.world, self.state, action)
        self.steps += 1
        # HRG-e 因果标注:
        # evidence_revealed_at = 首次读到真证据便签 (Note.is_true) 的回合;
        # decisive_mistake_at = 证据已揭示后, 首次把已满足的规则条件亲手破坏
        #   (vault_openable True→False 由本动作造成) 的回合。v1 限定: 只标注
        #   "破坏已满足条件" 型失误, 不标注 "持有证据却不行动" (字段审计注记)。
        if valid and action.startswith("read ") and self._evidence_revealed_at is None:
            note_name = action[len("read "):]
            note = next((n for n in self.world.notes if n.name == note_name), None)
            if note is not None and note.is_true:
                self._evidence_revealed_at = self.steps
        if valid and self._evidence_revealed_at is not None \
                and self._decisive_mistake_at is None \
                and pre_openable and not vault_openable(self.world, self.state):
            self._decisive_mistake_at = self.steps
        won = self.state.vault_open
        done = won or self.steps >= self.config.max_steps
        reward = VAULT_REWARD if won else 0.0
        obs = render_observation(self.world, self.state, feedback, self.config,
                                 rng=self._noise_rng)
        return obs, reward, done, self._info(action_valid=valid)

    def admissible_actions(self) -> List[str]:
        return admissible_actions(self.world, self.state)

    def _info(self, action_valid: bool) -> Dict:
        world, state = self.world, self.state
        return {
            "won": state.vault_open,
            "is_action_valid": action_valid,
            "steps": self.steps,
            "oracle_steps": self._oracle_steps,
            # C-sweep: coverage_level<1 时为 (字段 mask, 实测 C); 否则 (None, None)
            "phi_mask": list(self._phi_mask) if self._phi_mask is not None else None,
            "phi_coverage": self._phi_coverage,
            # HRG-e: 归因偏差分析用 ("预测得分骤降点 vs 真实因果点", 设计 §3)
            "causal_turns": {
                "evidence_revealed_at": self._evidence_revealed_at,
                "decisive_mistake_at": self._decisive_mistake_at,
            },
            "rule_family": world.rule.family,
            "rule_text": world.rule.describe(world.state_word),
            # 特权信息 (只进 info,绝不进观测): belief 探针 / MI / 归因用
            "latent_state": {
                "room": state.room,
                "device_states": dict(zip((d.name for d in world.stateful_devices),
                                          state.device_states)),
                "rule_progress": {"seq_progress": state.progress.seq_progress,
                                  "op_count": state.progress.op_count,
                                  "latched": state.progress.latched},
                "vault_openable": vault_openable(world, state),
                "inventory": sorted(state.inventory),
                "notes_read": sorted(state.notes_read),
                "vault_room": world.vault_room,
            },
        }
