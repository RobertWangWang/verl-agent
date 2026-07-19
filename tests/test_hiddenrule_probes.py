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

"""HRG-e 探针与因果标注测试 (P01)"""

import random

import pytest

from agent_system.environments.env_package.hiddenrule.hiddenrule.env import (
    HiddenRuleEnv,
    vault_openable,
)
from agent_system.environments.env_package.hiddenrule.hiddenrule.oracle import solve
from agent_system.environments.env_package.hiddenrule.hiddenrule.probes import (
    audit_no_leakage,
    generate_probes,
    rule_relevant_devices,
    score_answer,
)
from agent_system.environments.env_package.hiddenrule.hiddenrule.world import (
    HRGConfig,
    generate_world,
)


class TestCausalTurns:
    def test_reset_clears(self):
        env = HiddenRuleEnv(HRGConfig(rule_families=['conj']))
        _, info = env.reset(seed=1)
        assert info['causal_turns'] == {'evidence_revealed_at': None,
                                        'decisive_mistake_at': None}

    def test_evidence_revealed_on_true_note(self):
        # 搜一个真便签就在起始房间的 seed
        for seed in range(80):
            env = HiddenRuleEnv(HRGConfig(rule_families=['conj']))
            _, info = env.reset(seed=seed)
            true_note = next(n for n in env.world.notes if n.is_true)
            if true_note.room == env.state.room:
                _, _, _, info = env.step(f"read {true_note.name}")
                assert info['causal_turns']['evidence_revealed_at'] == 1
                return
        pytest.skip("80 seeds 内无起始房真便签 (布局分布问题, 非功能缺陷)")

    def test_distractor_note_does_not_reveal(self):
        for seed in range(80):
            env = HiddenRuleEnv(HRGConfig(rule_families=['conj'], n_distractor_notes=2))
            _, _ = env.reset(seed=seed)
            fake = next((n for n in env.world.notes
                         if not n.is_true and n.room == env.state.room), None)
            if fake is not None:
                _, _, _, info = env.step(f"read {fake.name}")
                assert info['causal_turns']['evidence_revealed_at'] is None
                return
        pytest.skip("80 seeds 内无起始房假便签")

    def test_oracle_path_no_decisive_mistake(self):
        """全知最优路径不应触发 decisive_mistake (不破坏已满足条件)"""
        env = HiddenRuleEnv(HRGConfig(rule_families=['conj'], n_rooms=4, n_devices=4))
        _, info = env.reset(seed=5)
        for action in solve(env.world):
            _, _, done, info = env.step(action)
        assert info['won'] is True
        assert info['causal_turns']['decisive_mistake_at'] is None

    def test_decisive_mistake_on_breaking_condition(self):
        """证据揭示后亲手破坏已满足条件 → 标注该回合"""
        for seed in range(200):
            cfg = HRGConfig(rule_families=['conj'], n_rooms=4, n_devices=4)
            env = HiddenRuleEnv(cfg)
            _, _ = env.reset(seed=seed)
            true_note = next(n for n in env.world.notes if n.is_true)
            if true_note.room != env.state.room:
                continue
            env.step(f"read {true_note.name}")  # 揭示证据
            solution = solve(env.world)
            if solution is None:
                continue
            # 走 oracle 路径直到 vault_openable, 但不开 vault
            opened = False
            for action in solution:
                if action == 'open vault':
                    opened = vault_openable(env.world, env.state)
                    break
                env.step(action)
            if not opened:
                continue
            # 找一个会破坏条件的机关动作 (conj: 把某个 required 设备拨走)
            relevant = rule_relevant_devices(env.world.rule)
            for action in env.admissible_actions():
                target = action.split(' ')[1] if action.startswith(('toggle', 'set')) else None
                if target in relevant:
                    _, _, _, info = env.step(action)
                    if not vault_openable(env.world, env.state):
                        assert info['causal_turns']['decisive_mistake_at'] == env.steps
                        return
        pytest.skip("200 seeds 内未构造出可破坏场景")


class TestProbes:
    def _world_state(self, seed=3):
        env = HiddenRuleEnv(HRGConfig(rule_families=['conj'], n_rooms=4, n_devices=4))
        env.reset(seed=seed)
        return env.world, env.state

    def test_truth_consistency(self):
        world, state = self._world_state()
        probes = generate_probes(world, state)
        for p in probes:
            if p['kind'] == 'vault_openable':
                assert p['answer'] == ('yes' if vault_openable(world, state) else 'no')
            if p['kind'] == 'rule_relevance':
                name = p['question'].split(' part of')[0].split()[-1]
                assert p['answer'] == ('yes' if name in rule_relevant_devices(world.rule) else 'no')

    def test_full_device_coverage(self):
        world, state = self._world_state()
        probes = generate_probes(world, state)
        for kind in ('device_state', 'rule_relevance'):
            asked = [p for p in probes if p['kind'] == kind]
            assert len(asked) == len(world.stateful_devices)

    def test_leakage_audit_passes_all_families(self):
        for fam in ('conj', 'seq', 'xor', 'count'):
            env = HiddenRuleEnv(HRGConfig(rule_families=[fam]))
            env.reset(seed=7)
            assert audit_no_leakage(generate_probes(env.world, env.state), env.world)

    def test_leakage_audit_catches_injected_leak(self):
        world, state = self._world_state()
        probes = generate_probes(world, state)
        rule_text = world.rule.describe(world.state_word)
        probes.append({'kind': 'rule_relevance',
                       'question': f"Hint: {rule_text}. Is lever_a relevant?",
                       'answer': 'yes'})
        with pytest.raises(AssertionError):
            audit_no_leakage(probes, world)

    def test_score_answer(self):
        assert score_answer('vault_openable', 'Yes, it would.', 'yes') == 1.0
        assert score_answer('vault_openable', 'no', 'yes') == 0.0
        assert score_answer('rule_relevance', 'I think no.', 'no') == 1.0
        assert score_answer('device_state', 'lever_a is up', 'up') == 1.0
        assert score_answer('device_state', 'it is set to 2', 'set to 2') == 1.0
        assert score_answer('device_state', 'down probably', 'up') == 0.0
        assert score_answer('vault_openable', '', 'yes') == 0.0


class TestHarnessSmoke:
    def test_random_backend_end_to_end(self):
        """无 GPU 冒烟: random 后端跑通 run_probing + aggregate"""
        from examples.hiddenrule.probe_checkpoint import aggregate, run_probing
        rng = random.Random(0)
        records = run_probing(
            generate_fn=lambda prompt: rng.choice(['yes', 'no', 'up', '1']),
            action_fn=lambda obs, admissible, history: rng.choice(admissible),
            n_episodes=2, probe_every=10,
            config=HRGConfig(rule_families=['conj'], n_rooms=4, n_devices=4, max_steps=20),
        )
        assert records and all('score' in r and 'causal_turns' in r for r in records)
        summary = aggregate(records)
        assert any(k.startswith('conj/') for k in summary)
