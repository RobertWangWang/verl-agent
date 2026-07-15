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
全知 BFS 解算器 (docs/hiddenrule_gym_design.md §3)

规则已知条件下的最短解,用于:
- 难度校准与分数归一化 (oracle_steps 进 info)
- HRG-a 验收: 任意采样 episode 必须可解且步数 <= horizon

与 env 共用 transition/admissible 纯函数,读便签与 look 对动力学无影响,剪掉。
"""

from collections import deque
from typing import List, Optional

from .env import CoreState, admissible_actions, initial_state, transition
from .world import World

MAX_NODES = 200_000


def _search_key(state: CoreState):
    # notes_read 不影响动力学,从搜索键中剔除以缩小空间
    return (state.room, state.device_states, state.progress,
            state.inventory, state.unlocked_doors, state.vault_open)


def solve(world: World, max_nodes: int = MAX_NODES) -> Optional[List[str]]:
    """返回最短动作序列 (含最后的 open vault);不可解或超预算返回 None。"""
    start = initial_state(world)
    queue = deque([(start, [])])
    seen = {_search_key(start)}
    expanded = 0

    while queue:
        state, path = queue.popleft()
        expanded += 1
        if expanded > max_nodes:
            return None
        for action in admissible_actions(world, state):
            if action == "look" or action.startswith("read "):
                continue
            nxt, _, valid = transition(world, state, action)
            if not valid:
                continue
            if nxt.vault_open:
                return path + [action]
            key = _search_key(nxt)
            if key not in seen:
                seen.add(key)
                queue.append((nxt, path + [action]))
    return None
