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
HiddenRule-Gym 世界模型与布局生成 (docs/hiddenrule_gym_design.md §1)

房间图 (生成树 + 1 条冗余边) + 机关 + 门 (open/key) + 便签 + 钥匙 + 保险库。
同一 seed 完全确定性地生成同一世界 (GRPO 组环境语义的基础)。
"""

import random
import string
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .rules import Rule, sample_fake_rule, sample_rule


@dataclass
class HRGConfig:
    n_rooms: int = 5
    n_devices: int = 6
    rule_families: tuple = ("conj", "seq", "xor", "count")
    rule_arity: int = 2
    evidence_lag: int = 2          # 真便签所在房间到保险库房间的图距离 (尽力满足)
    n_distractor_notes: int = 1
    max_steps: int = 60
    key_door_prob: float = 0.5     # 通往保险库路径上出现钥匙门的概率
    # ---- HRG-b 旋钮 (本阶段占位,渲染层实现) ----
    p_obs: float = 1.0
    obs_flip_prob: float = 0.0
    n_sensor_channels: int = 0
    coverage_level: float = 1.0


@dataclass
class Device:
    name: str
    kind: str        # 'lever' | 'dial' | 'button'
    room: int
    n_states: int    # lever=2, dial>=3, button=1 (无状态,事件源)

    @property
    def stateful(self) -> bool:
        return self.kind != "button"


@dataclass
class Door:
    room_a: int
    room_b: int
    kind: str                      # 'open' | 'key'
    key_name: Optional[str] = None

    def other(self, room: int) -> int:
        return self.room_b if room == self.room_a else self.room_a

    def connects(self, room: int) -> bool:
        return room in (self.room_a, self.room_b)


@dataclass
class Note:
    name: str
    room: int
    text: str
    is_true: bool


@dataclass
class Item:
    name: str
    room: int


@dataclass
class World:
    config: HRGConfig
    n_rooms: int
    start_room: int
    vault_room: int
    doors: List[Door]
    devices: List[Device]
    notes: List[Note]
    items: List[Item]
    rule: Rule
    # 派生索引
    stateful_devices: List[Device] = field(default_factory=list)
    stateful_index: Dict[str, int] = field(default_factory=dict)
    device_by_name: Dict[str, Device] = field(default_factory=dict)

    def __post_init__(self):
        self.stateful_devices = [d for d in self.devices if d.stateful]
        self.stateful_index = {d.name: i for i, d in enumerate(self.stateful_devices)}
        self.device_by_name = {d.name: d for d in self.devices}

    def doors_from(self, room: int) -> List[Door]:
        return [d for d in self.doors if d.connects(room)]

    def state_word(self, device_name: str, state: int) -> str:
        """机关状态的自然语言 (便签/观测共用,保证语言一致可解析)"""
        dev = self.device_by_name[device_name]
        if dev.kind == "lever":
            return "up" if state == 1 else "down"
        return f"set to {state}"

    def distances_from(self, room: int) -> List[int]:
        """全图 BFS 距离 (忽略门锁,布局层面的图距离)"""
        dist = [-1] * self.n_rooms
        dist[room] = 0
        queue = deque([room])
        while queue:
            cur = queue.popleft()
            for door in self.doors_from(cur):
                nxt = door.other(cur)
                if dist[nxt] < 0:
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)
        return dist


def _device_kinds(cfg: HRGConfig) -> List[str]:
    """机关种类分配: 保证 xor/conj 有足够的 lever,seq/count 有 button。"""
    n_levers = max(2, cfg.rule_arity + 1)
    kinds = ["lever"] * min(n_levers, cfg.n_devices)
    toggle = True
    while len(kinds) < cfg.n_devices:
        kinds.append("dial" if toggle else "button")
        toggle = not toggle
    return kinds


def generate_world(cfg: HRGConfig, seed: int) -> World:
    rng = random.Random(seed)

    # 1. 房间图: 生成树 + 至多 1 条冗余边
    parents = [None] + [rng.randrange(0, i) for i in range(1, cfg.n_rooms)]
    edges = {(min(i, p), max(i, p)) for i, p in enumerate(parents) if p is not None}
    tree_edges = set(edges)
    candidates = [(a, b) for a in range(cfg.n_rooms) for b in range(a + 1, cfg.n_rooms)
                  if (a, b) not in edges]
    if candidates:
        edges.add(rng.choice(candidates))
    doors = [Door(a, b, "open") for a, b in sorted(edges)]

    # 2. 保险库: 距起点最远的房间
    tmp_world = World(cfg, cfg.n_rooms, 0, 0, doors, [], [], [],
                      Rule(family="count", device="x", n=2))
    dist_from_start = tmp_world.distances_from(0)
    vault_room = max(range(cfg.n_rooms), key=lambda r: dist_from_start[r])

    # 3. 机关
    kinds = _device_kinds(cfg)
    rng.shuffle(kinds)
    devices = []
    for i, kind in enumerate(kinds):
        letter = string.ascii_uppercase[i]
        n_states = 2 if kind == "lever" else (rng.choice([3, 4]) if kind == "dial" else 1)
        devices.append(Device(name=f"{kind}_{letter}", kind=kind,
                              room=rng.randrange(cfg.n_rooms), n_states=n_states))

    # 4. 隐藏规则
    stateful = [d for d in devices if d.stateful]
    buttons = [d for d in devices if not d.stateful]
    family = rng.choice(tuple(cfg.rule_families))
    rule = sample_rule(family, stateful, buttons, cfg.rule_arity, rng)

    world = World(cfg, cfg.n_rooms, 0, vault_room, doors, devices, [], [], rule)

    # 5. 钥匙门: 从起点到保险库的树路径上抽一条边加锁,钥匙放在起点一侧
    if rng.random() < cfg.key_door_prob:
        path_edges = _tree_path_edges(parents, 0, vault_room)
        path_edges = [e for e in path_edges if e in tree_edges]
        if path_edges:
            lock_edge = rng.choice(path_edges)
            start_side = _component_without_edge(cfg.n_rooms, edges, lock_edge, 0)
            for door in world.doors:
                if (min(door.room_a, door.room_b), max(door.room_a, door.room_b)) == lock_edge:
                    # 若冗余边绕过了该门则锁无意义,可接受 (oracle 仍最短路)
                    door.kind = "key"
                    door.key_name = "brass_key"
                    world.items.append(Item(name="brass_key", room=rng.choice(sorted(start_side))))
                    break

    # 6. 便签: 真便签按 evidence_lag 放置,假便签随机
    dist_from_vault = world.distances_from(vault_room)
    target_lag = max(1, min(cfg.evidence_lag, max(dist_from_vault)))
    lag_rooms = [r for r in range(cfg.n_rooms) if dist_from_vault[r] == target_lag]
    true_room = rng.choice(lag_rooms) if lag_rooms else world.start_room
    world.notes.append(Note(name="note_1", room=true_room,
                            text=rule.describe(world.state_word), is_true=True))
    for j in range(cfg.n_distractor_notes):
        fake_text = sample_fake_rule(rule, stateful, buttons, cfg.rule_arity, rng,
                                     world.state_word)
        world.notes.append(Note(name=f"note_{j + 2}", room=rng.randrange(cfg.n_rooms),
                                text=fake_text, is_true=False))
    return world


def _tree_path_edges(parents, a, b):
    """生成树上 a→b 路径的边集"""
    def path_to_root(x):
        path = [x]
        while parents[path[-1]] is not None:
            path.append(parents[path[-1]])
        return path

    pa, pb = path_to_root(a), path_to_root(b)
    common = set(pa) & set(pb)
    lca = next(x for x in pa if x in common)
    edges = []
    for path in (pa, pb):
        for node in path:
            if node == lca:
                break
            edges.append((min(node, parents[node]), max(node, parents[node])))
    return edges


def _component_without_edge(n_rooms, edges, removed, root):
    """去掉一条边后 root 所在的连通分量"""
    adj = {r: [] for r in range(n_rooms)}
    for a, b in edges:
        if (a, b) == removed:
            continue
        adj[a].append(b)
        adj[b].append(a)
    seen = {root}
    queue = deque([root])
    while queue:
        cur = queue.popleft()
        for nxt in adj[cur]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen
