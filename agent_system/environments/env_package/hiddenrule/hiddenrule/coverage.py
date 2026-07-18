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
特征覆盖度 C 的精确计算 (docs/hiddenrule_gym_design.md §2.1, 提案 v0.2 §4.1 协议第 3 条)

C = I(Φ(o); s) / H(s),s 均匀分布在可达状态集上。
- 无噪声时 Φ 是 s 的确定函数 → I(Φ; s) = H(Φ(s)),精确枚举;
- 有观测翻转噪声时 I = H(Φ) − H(Φ|s),对每个 s 的翻转分布做 Monte Carlo。

字段词表 (schema 级,与观测渲染一一对应):
    'room'                — agent 所在房间
    'device:<name>'       — 某个有状态机关的读数
    'door:<a>-<b>'        — 某扇门的开/锁显示
预置 coverage ladder 通过贪心加字段校准到目标 C (calibrate_masks)。
"""

import math
import random
from collections import Counter, deque
from typing import Dict, FrozenSet, List, Optional, Tuple

from .env import CoreState, admissible_actions, initial_state, transition
from .world import World

MAX_STATES = 60_000


def enumerate_reachable(world: World, cap: int = MAX_STATES) -> List[CoreState]:
    """BFS 枚举可达状态 (含 notes_read 折叠,与 oracle 同键)"""
    start = initial_state(world)

    def key(s: CoreState):
        return (s.room, s.device_states, s.progress, s.inventory,
                s.unlocked_doors, s.vault_open)

    seen = {key(start)}
    states = [start]
    queue = deque([start])
    while queue and len(states) < cap:
        state = queue.popleft()
        for action in admissible_actions(world, state):
            if action == "look" or action.startswith("read "):
                continue
            nxt, _, valid = transition(world, state, action)
            if not valid:
                continue
            k = key(nxt)
            if k not in seen:
                seen.add(k)
                states.append(nxt)
                queue.append(nxt)
    return states


def all_fields(world: World) -> List[str]:
    fields = ['room']
    fields += [f'device:{d.name}' for d in world.stateful_devices]
    fields += [f'door:{min(d.room_a, d.room_b)}-{max(d.room_a, d.room_b)}'
               for d in world.doors if d.kind == 'key']
    return fields


def _field_value(world: World, state: CoreState, field: str,
                 flip_prob: float = 0.0, rng: Optional[random.Random] = None):
    """某字段在观测里的读数;flip_prob>0 时按观测噪声模型翻转"""
    if field == 'room':
        return state.room  # 位置不加噪 (设计: 噪声只作用于机关/门读数)
    if field.startswith('device:'):
        name = field.split(':', 1)[1]
        dev = world.device_by_name[name]
        value = state.device_states[world.stateful_index[name]]
        if flip_prob > 0 and rng is not None and rng.random() < flip_prob:
            others = [v for v in range(dev.n_states) if v != value]
            value = rng.choice(others)
        return value
    if field.startswith('door:'):
        a, b = field.split(':', 1)[1].split('-')
        a, b = int(a), int(b)
        for i, door in enumerate(world.doors):
            if {door.room_a, door.room_b} == {a, b}:
                unlocked = door.kind == 'open' or i in state.unlocked_doors
                if flip_prob > 0 and rng is not None and rng.random() < flip_prob:
                    unlocked = not unlocked
                return unlocked
    raise ValueError(f"unknown field {field}")


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)


def _state_key(state: CoreState):
    return (state.room, state.device_states, state.progress,
            state.inventory, state.unlocked_doors, state.vault_open)


def coverage(world: World, mask: FrozenSet[str],
             states: Optional[List[CoreState]] = None,
             flip_prob: float = 0.0, mc_samples: int = 32,
             seed: int = 0) -> float:
    """
    C = I(Φ_mask(o); s) / H(s),s 均匀分布于可达状态。

    flip_prob=0: Φ 是 s 的确定函数,I = H(Φ(s)),精确;
    flip_prob>0: 每状态 mc_samples 次采样估计 H(Φ) 与 H(Φ|s)。
    """
    if states is None:
        states = enumerate_reachable(world)
    # 覆盖度定义在非终止可达状态上: vault_open 是吸收态,episode 已结束,
    # 无未来可预测;保留它会与同观测的非终止态碰撞,人为压低 C。
    states = [s for s in states if not s.vault_open]
    n = len(states)
    if n <= 1:
        return 1.0
    h_s = math.log2(n)  # s 均匀 → H(s) = log2 |S|

    fields = sorted(mask)
    if flip_prob <= 0:
        phi_counter = Counter(
            tuple(_field_value(world, s, f) for f in fields) for s in states
        )
        return _entropy(phi_counter) / h_s

    rng = random.Random(seed)
    phi_counter: Counter = Counter()
    h_phi_given_s = 0.0
    for state in states:
        per_state: Counter = Counter()
        for _ in range(mc_samples):
            obs = tuple(_field_value(world, state, f, flip_prob, rng) for f in fields)
            per_state[obs] += 1
            phi_counter[obs] += 1
        h_phi_given_s += _entropy(per_state)
    h_phi_given_s /= n
    mi = max(0.0, _entropy(phi_counter) - h_phi_given_s)
    return min(1.0, mi / h_s)


def sweep_fields(world: World) -> List[str]:
    """
    C-sweep 的字段池: room + 有状态机关读数。
    门字段被排除 —— predict 块没有对应的可奖励特征 (design doc §2.1: 测得的 C
    必须定义在与预测奖励相同的 Φ 家族上,否则 x 轴失真)。
    """
    return ['room'] + [f'device:{d.name}' for d in world.stateful_devices]


def calibrate_masks(world: World,
                    targets: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
                    states: Optional[List[CoreState]] = None,
                    fields: Optional[List[str]] = None) -> Dict[float, FrozenSet[str]]:
    """
    贪心构造 coverage ladder: 从空集起,每次加入使 C 增益最大的字段,
    首次达到各目标档位时记录当时的 mask。保证 ladder 单调。
    fields 限定候选字段池 (默认全字段; C-sweep 用 sweep_fields)。
    """
    if states is None:
        states = enumerate_reachable(world)
    remaining = set(fields) if fields is not None else set(all_fields(world))
    mask: set = set()
    ladder: Dict[float, FrozenSet[str]] = {}
    current_c = 0.0
    targets_left = sorted(targets)

    while targets_left and remaining:
        best_field, best_c = None, current_c
        for field in sorted(remaining):
            c = coverage(world, frozenset(mask | {field}), states=states)
            if c > best_c:
                best_field, best_c = field, c
        if best_field is None:  # 加任何字段都无增益 (剩余字段冗余)
            break
        mask.add(best_field)
        remaining.discard(best_field)
        current_c = best_c
        while targets_left and current_c >= targets_left[0] - 1e-9:
            ladder[targets_left.pop(0)] = frozenset(mask)

    # 达不到的高档位 (如 seq/count 的 latch 本质不可观测) 用池内全字段兜底
    full = frozenset(fields) if fields is not None else frozenset(all_fields(world))
    for t in targets_left:
        ladder[t] = full
    return ladder
