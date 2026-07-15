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
观测渲染: latent state → 结构化文本 (docs/hiddenrule_gym_design.md §1.4, §2.2-2.3)

HRG-b 旋钮 (全部作用于文本层,latent state 与动作语义不受影响):
- p_obs < 1.0        : 本房间的机关/门/物品字段以 (1−p_obs) 概率被隐藏 (部分可观测)
- obs_flip_prob > 0  : 机关读数/门状态以该概率显示错误值 (aleatoric 观测噪声)
- n_sensor_channels>0: 追加纯随机的传感器面板行 (noisy-TV 干扰通道)

噪声由 env 传入的专用 rng 驱动,同 seed 同动作序列 → 逐字节相同的观测 (确定性回放)。
"""

import random
from typing import Optional


def render_observation(world, state, feedback: str, config,
                       rng: Optional[random.Random] = None) -> str:
    p_obs = getattr(config, 'p_obs', 1.0)
    flip = getattr(config, 'obs_flip_prob', 0.0)
    n_sensors = getattr(config, 'n_sensor_channels', 0)
    noisy = rng is not None and (p_obs < 1.0 or flip > 0.0 or n_sensors > 0)

    def visible() -> bool:
        return not (noisy and p_obs < 1.0 and rng.random() >= p_obs)

    def maybe_flip_device(dev, true_value: int) -> int:
        if noisy and flip > 0.0 and rng.random() < flip:
            others = [v for v in range(dev.n_states) if v != true_value]
            return rng.choice(others)
        return true_value

    def maybe_flip_bool(value: bool) -> bool:
        if noisy and flip > 0.0 and rng.random() < flip:
            return not value
        return value

    room = state.room
    hidden_something = False
    lines = [f"Last action result: {feedback}"]
    lines.append(f"You are in room {room + 1} of {world.n_rooms}. "
                 f"The vault is in room {world.vault_room + 1}.")

    devices_here = [d for d in world.devices if d.room == room]
    if devices_here:
        parts = []
        for dev in devices_here:
            if not visible():
                hidden_something = True
                continue
            if dev.stateful:
                value = maybe_flip_device(
                    dev, state.device_states[world.stateful_index[dev.name]])
                parts.append(f"{dev.name} is {world.state_word(dev.name, value)}")
            else:
                parts.append(f"{dev.name} (a button)")
        if parts:
            lines.append("Devices here: " + "; ".join(parts) + ".")

    door_parts = []
    for i, door in enumerate(world.doors):
        if not door.connects(room):
            continue
        if not visible():
            hidden_something = True
            continue
        target = door.other(room) + 1
        unlocked = door.kind == "open" or i in state.unlocked_doors
        if door.kind == "key":
            unlocked = maybe_flip_bool(unlocked)
        if unlocked:
            door_parts.append(f"room {target} (open)")
        else:
            door_parts.append(f"room {target} (locked, needs {door.key_name})")
    if door_parts:
        lines.append("Doors from here lead to: " + ", ".join(door_parts) + ".")

    candidates = [n.name for n in world.notes if n.room == room]
    candidates += [it.name for it in world.items
                   if it.room == room and it.name not in state.inventory]
    seen = []
    for name in candidates:
        if visible():
            seen.append(name)
        else:
            hidden_something = True
    if seen:
        lines.append("You see: " + ", ".join(seen) + ".")

    lines.append("Inventory: [" + ", ".join(sorted(state.inventory)) + "].")

    if state.room == world.vault_room:
        lines.append("The vault is here." + (" It stands open." if state.vault_open
                                             else " It is sealed."))

    if hidden_something:
        lines.append("The room is dimly lit; you cannot make out everything.")

    if noisy and n_sensors > 0:
        readings = "; ".join(f"P{k + 1}={rng.random():.2f}" for k in range(n_sensors))
        lines.append(f"Sensor panel: {readings}.")  # 纯噪声,与 latent 无关 (noisy-TV)

    return "\n".join(lines)
