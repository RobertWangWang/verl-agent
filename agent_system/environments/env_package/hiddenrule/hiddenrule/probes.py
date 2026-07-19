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
HRG-e 信念探针 (P01, docs/hiddenrule_gym_design.md §3 + 提案 v0.2 探针协议)

对训好的策略 ckpt 离线提问, 用特权 latent 判分, 回答 "策略实际维护了哪些信念":
- device_state: 当前机关读数 (可从观测记住 → 记忆探针)
- vault_openable: 现在开保险库会不会成功 (需要规则 belief → 规则探针)
- rule_relevance: 某机关是否参与隐藏机制 (纯规则 belief)

防泄漏协议: 问题模板对**全部**机关逐一提问 (问题集合本身不暴露哪些相关),
且 audit_no_leakage 断言问题文本不含规则描述 (rule_text) 的任何独有词串。
"""

import re
from typing import Any, Dict, List

from .env import CoreState, _device_state_of, vault_openable
from .world import World

_YES = {'yes', 'y', 'true'}
_NO = {'no', 'n', 'false'}


def rule_relevant_devices(rule) -> set:
    """各规则族的相关机关名集合 (特权信息, 仅探针判分用)"""
    if rule.family == 'conj':
        return {name for name, _ in rule.conditions}
    if rule.family == 'xor':
        return set(rule.devices)
    if rule.family == 'seq':
        return set(rule.sequence)
    if rule.family == 'count':
        return {rule.device}
    raise ValueError(f"unknown family {rule.family}")


def generate_probes(world: World, state: CoreState,
                    kinds=('device_state', 'vault_openable', 'rule_relevance')) -> List[Dict[str, Any]]:
    """生成 (question, answer, kind) 探针列表; answer 来自纯函数/特权 latent。"""
    probes: List[Dict[str, Any]] = []
    relevant = rule_relevant_devices(world.rule)
    if 'device_state' in kinds:
        for dev in world.stateful_devices:
            value = _device_state_of(world, state, dev.name)
            probes.append({
                'kind': 'device_state',
                'question': f"What is the current state of {dev.name}? "
                            f"Answer with just the state word or number.",
                'answer': world.state_word(dev.name, value),
            })
    if 'vault_openable' in kinds:
        probes.append({
            'kind': 'vault_openable',
            'question': "If you tried 'open vault' right now, would it open? "
                        "Answer yes or no.",
            'answer': 'yes' if vault_openable(world, state) else 'no',
        })
    if 'rule_relevance' in kinds:
        for dev in world.stateful_devices:
            probes.append({
                'kind': 'rule_relevance',
                'question': f"Is {dev.name} part of the hidden mechanism that "
                            f"controls the vault? Answer yes or no.",
                'answer': 'yes' if dev.name in relevant else 'no',
            })
    return probes


def audit_no_leakage(probes: List[Dict[str, Any]], world: World) -> bool:
    """
    字段审计 (HRG-e 验收标准): 探针问题不得泄漏规则内容。
    - 问题文本不含 rule_text 的独有词串 (机关名与通用词除外);
    - device 类探针必须覆盖全部机关 (问题集合的构成不暴露相关性)。
    违规抛 AssertionError。
    """
    rule_text = world.rule.describe(world.state_word).lower()
    device_names = {d.name.lower() for d in world.stateful_devices}
    # conj 等族的规则文本全由通用词+机关名组成 (无独有词) —— 用连续 n-gram
    # 窗口检测: 规则文本的任何连续 5 词片段出现在问题里即判泄漏
    rule_tokens = re.findall(r'[a-z][a-z_0-9]*', rule_text)
    n = min(5, max(2, len(rule_tokens)))
    spans = {' '.join(rule_tokens[i:i + n]) for i in range(len(rule_tokens) - n + 1)}
    for probe in probes:
        q = ' '.join(re.findall(r'[a-z][a-z_0-9]*', probe['question'].lower()))
        for span in spans:
            assert span not in q, \
                f"泄漏审计失败: 探针问题含规则文本片段 '{span}': {probe['question']}"
    for kind in ('device_state', 'rule_relevance'):
        asked = {p['question'] for p in probes if p['kind'] == kind}
        if asked:
            for d in device_names:
                assert any(d in q.lower() for q in asked), \
                    f"覆盖审计失败: {kind} 探针未覆盖机关 {d} (选择性提问会泄漏相关性)"
    return True


def score_answer(kind: str, model_text: str, answer: str) -> float:
    """规则化判分: yes/no 探针取首个 yes/no token; 状态探针做子串匹配。0/1。"""
    if not model_text:
        return 0.0
    text = model_text.strip().lower()
    if kind in ('vault_openable', 'rule_relevance'):
        m = re.search(r'\b(yes|no|y|n|true|false)\b', text)
        if m is None:
            return 0.0
        said_yes = m.group(1) in _YES
        return 1.0 if said_yes == (answer == 'yes') else 0.0
    # device_state: 答案词 (如 'up' / 'set to 2' / '2') 子串匹配
    ans = answer.lower()
    if ans in text:
        return 1.0
    num = re.search(r'\d+', ans)
    if num and re.search(rf'\b{num.group(0)}\b', text):
        return 1.0
    return 0.0
