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

import re
from typing import List


def hiddenrule_projection(actions: List[str], action_pools: List[List[str]],
                          require_think: bool = True):
    """
    <action>...</action> 解析 (与 alfworld_projection 同语义,含 require_think 开关
    —— Qwen3 + enable_thinking=False 时必须设 False,见 alfworld/projection.py 注释)。
    """
    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]
        actions[i] = actions[i].lower()

        start_idx = actions[i].find("<action>")
        end_idx = actions[i].find("</action>")
        if start_idx == -1 or end_idx == -1:
            actions[i] = actions[i][-30:]
            continue

        actions[i] = actions[i][start_idx + len("<action>"):end_idx].strip().lower()
        valids[i] = 1

        if require_think:
            if original_str.find("<think>") == -1 or original_str.find("</think>") == -1:
                valids[i] = 0

        if re.search(r'[一-鿿]', original_str):
            valids[i] = 0

    return actions, valids
