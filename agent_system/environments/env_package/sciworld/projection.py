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


def sciworld_projection(actions: List[str], require_think: bool = True):
    """<action>...</action> 抽取 (与 webshop_projection 同构)。

    ScienceWorld 服务端契约: 无效动作 = no-op ("No known action matches that
    input."), 因此这里只做格式合法性判定, 不做动作词表校验。
    require_think: Qwen3 + enable_thinking=False 场景必须置 False (见 webshop 注)。
    """
    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]
        actions[i] = actions[i].lower()

        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            actions[i] = actions[i][-20:]
            continue

        actions[i] = actions[i][start_idx + len(start_tag):end_idx].strip()
        valids[i] = 1

        if require_think:
            if original_str.find("<think>") == -1 or original_str.find("</think>") == -1:
                valids[i] = 0

        if re.search(r'[一-鿿]', original_str):
            valids[i] = 0

    return actions, valids
