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
观测渲染: latent state → 结构化文本 (docs/hiddenrule_gym_design.md §1.4)

HRG-a: 完整无噪渲染。
HRG-b 待实装 (旋钮已在 HRGConfig 占位): p_obs 字段隐藏、obs_flip_prob 状态翻转、
n_sensor_channels 纯噪声传感器面板 (noisy-TV 通道)。
"""


def render_observation(world, state, feedback: str, config) -> str:
    room = state.room
    lines = [f"Last action result: {feedback}"]
    lines.append(f"You are in room {room + 1} of {world.n_rooms}. "
                 f"The vault is in room {world.vault_room + 1}.")

    devices_here = [d for d in world.devices if d.room == room]
    if devices_here:
        parts = []
        for dev in devices_here:
            if dev.stateful:
                word = world.state_word(dev.name, state.device_states[world.stateful_index[dev.name]])
                parts.append(f"{dev.name} is {word}")
            else:
                parts.append(f"{dev.name} (a button)")
        lines.append("Devices here: " + "; ".join(parts) + ".")

    door_parts = []
    for i, door in enumerate(world.doors):
        if not door.connects(room):
            continue
        target = door.other(room) + 1
        if door.kind == "open" or i in state.unlocked_doors:
            door_parts.append(f"room {target} (open)")
        else:
            door_parts.append(f"room {target} (locked, needs {door.key_name})")
    lines.append("Doors from here lead to: " + ", ".join(door_parts) + ".")

    visible = [n.name for n in world.notes if n.room == room]
    visible += [it.name for it in world.items
                if it.room == room and it.name not in state.inventory]
    if visible:
        lines.append("You see: " + ", ".join(visible) + ".")

    lines.append("Inventory: [" + ", ".join(sorted(state.inventory)) + "].")

    if state.room == world.vault_room:
        lines.append("The vault is here." + (" It stands open." if state.vault_open
                                             else " It is sealed."))
    return "\n".join(lines)
