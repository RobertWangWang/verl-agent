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
HiddenRule-Gym HRG-a 验收测试 (docs/hiddenrule_gym_design.md §5)

- 四族规则 × 100 随机 episode: oracle 全部可解且步数 <= horizon
- 同种子完全确定性回放
- 各规则族/机制的手工语义用例

运行: pytest tests/test_hiddenrule_core.py -v
"""

import pytest

from agent_system.environments.env_package.hiddenrule.hiddenrule import (
    Device,
    Door,
    HiddenRuleEnv,
    HRGConfig,
    Rule,
    World,
    admissible_actions,
    initial_state,
    solve,
    transition,
)

RULE_FAMILIES = ("conj", "seq", "xor", "count")
N_SEEDS = 100


# ---------------------------------------------------------------------------
# 验收 1: oracle 可解性 (四族 × 100 seed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", RULE_FAMILIES)
def test_oracle_solves_all_episodes(family):
    cfg = HRGConfig(rule_families=(family,))
    env = HiddenRuleEnv(cfg)
    for seed in range(N_SEEDS):
        _, info = env.reset(seed=seed)
        assert info["rule_family"] == family
        assert info["oracle_steps"] > 0, f"seed {seed}: unsolvable"
        assert info["oracle_steps"] <= cfg.max_steps, \
            f"seed {seed}: oracle needs {info['oracle_steps']} > horizon {cfg.max_steps}"


@pytest.mark.parametrize("family", RULE_FAMILIES)
def test_oracle_path_actually_wins(family):
    """oracle 给出的动作序列在 env 里真实回放必须获胜 (env/oracle 语义一致性)"""
    cfg = HRGConfig(rule_families=(family,))
    env = HiddenRuleEnv(cfg)
    for seed in range(0, N_SEEDS, 10):
        env.reset(seed=seed)
        path = solve(env.world)
        assert path is not None
        total_reward, done = 0.0, False
        for action in path:
            assert not done, f"seed {seed}: episode ended before path exhausted"
            _, reward, done, info = env.step(action)
            assert info["is_action_valid"], f"seed {seed}: oracle action invalid: {action}"
            total_reward += reward
        assert info["won"], f"seed {seed}: oracle path did not open vault"
        assert total_reward == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 验收 2: 确定性
# ---------------------------------------------------------------------------

def test_same_seed_same_world_and_replay():
    cfg = HRGConfig()
    env1, env2 = HiddenRuleEnv(cfg), HiddenRuleEnv(cfg)
    for seed in (0, 7, 42):
        obs1, info1 = env1.reset(seed=seed)
        obs2, info2 = env2.reset(seed=seed)
        assert obs1 == obs2
        assert info1["rule_text"] == info2["rule_text"]
        assert info1["oracle_steps"] == info2["oracle_steps"]
        path = solve(env1.world)
        for action in path:
            o1, r1, d1, i1 = env1.step(action)
            o2, r2, d2, i2 = env2.step(action)
            assert o1 == o2 and r1 == r2 and d1 == d2
            assert i1["latent_state"] == i2["latent_state"]


def test_different_seeds_differ():
    env = HiddenRuleEnv(HRGConfig())
    texts = set()
    for seed in range(20):
        _, info = env.reset(seed=seed)
        texts.add(info["rule_text"])
    assert len(texts) > 5, "20 个种子的规则几乎相同,采样多样性不足"


# ---------------------------------------------------------------------------
# 验收 3: 手工语义用例 (固定 fixture world,不走随机生成)
# ---------------------------------------------------------------------------

def make_fixture_world(rule: Rule) -> World:
    """3 房间线形图: 0 -- 1 -- 2(vault)。机关都在房间 0,便签在房间 1。"""
    cfg = HRGConfig(n_rooms=3, max_steps=60)
    devices = [
        Device(name="lever_A", kind="lever", room=0, n_states=2),
        Device(name="lever_B", kind="lever", room=0, n_states=2),
        Device(name="dial_C", kind="dial", room=0, n_states=3),
        Device(name="button_D", kind="button", room=0, n_states=1),
    ]
    doors = [Door(0, 1, "open"), Door(1, 2, "open")]
    world = World(config=cfg, n_rooms=3, start_room=0, vault_room=2,
                  doors=doors, devices=devices, notes=[], items=[], rule=rule)
    return world


class TestConjSemantics:
    def setup_method(self):
        rule = Rule(family="conj", conditions=(("lever_A", 1), ("dial_C", 2)))
        self.world = make_fixture_world(rule)
        self.state = initial_state(self.world)

    def run(self, *actions):
        for a in actions:
            self.state, feedback, valid = transition(self.world, self.state, a)
            assert valid, f"unexpected invalid action: {a}"
        return feedback

    def test_vault_sealed_until_condition(self):
        self.run("go to room 2", "go to room 3")
        feedback = self.run("open vault")
        assert "does not budge" in feedback
        assert not self.state.vault_open

    def test_condition_met_opens(self):
        self.run("toggle lever_A", "set dial_C to 2", "go to room 2", "go to room 3")
        self.run("open vault")
        assert self.state.vault_open

    def test_condition_is_dynamic(self):
        """状态谓词类: 条件被破坏后保险库重新打不开"""
        self.run("toggle lever_A", "set dial_C to 2", "toggle lever_A",  # 破坏
                 "go to room 2", "go to room 3")
        feedback = self.run("open vault")
        assert "does not budge" in feedback


class TestXorSemantics:
    def setup_method(self):
        rule = Rule(family="xor", devices=("lever_A", "lever_B"), parity=1)
        self.world = make_fixture_world(rule)
        self.state = initial_state(self.world)

    def run(self, *actions):
        for a in actions:
            self.state, _, valid = transition(self.world, self.state, a)
            assert valid

    def test_odd_parity(self):
        self.run("toggle lever_A", "go to room 2", "go to room 3", "open vault")
        assert self.state.vault_open  # 1 个 up = 奇数

    def test_even_parity_fails(self):
        self.run("toggle lever_A", "toggle lever_B",  # 2 个 up = 偶数
                 "go to room 2", "go to room 3", "open vault")
        assert not self.state.vault_open


class TestSeqSemantics:
    def setup_method(self):
        rule = Rule(family="seq", sequence=("lever_A", "button_D"))
        self.world = make_fixture_world(rule)
        self.state = initial_state(self.world)

    def run(self, *actions):
        for a in actions:
            self.state, _, valid = transition(self.world, self.state, a)
            assert valid

    def test_correct_order_latches(self):
        self.run("toggle lever_A", "press button_D")
        assert self.state.progress.latched
        # 锁存后破坏 lever 状态也不影响
        self.run("toggle lever_A", "go to room 2", "go to room 3", "open vault")
        assert self.state.vault_open

    def test_wrong_order_resets(self):
        self.run("press button_D", "press button_D")  # 未从 lever_A 开始
        assert not self.state.progress.latched
        assert self.state.progress.seq_progress == 0

    def test_wrong_op_midway_restarts(self):
        self.run("toggle lever_A", "toggle lever_B")  # 中途走错
        assert self.state.progress.seq_progress == 0
        self.run("toggle lever_A", "press button_D")  # 重来
        assert self.state.progress.latched


class TestCountSemantics:
    def setup_method(self):
        rule = Rule(family="count", device="button_D", n=3)
        self.world = make_fixture_world(rule)
        self.state = initial_state(self.world)

    def test_latch_at_n(self):
        for i in range(3):
            assert not self.state.progress.latched
            self.state, _, _ = transition(self.world, self.state, "press button_D")
        assert self.state.progress.latched
        assert self.state.progress.op_count == 3


class TestKeyDoor:
    def setup_method(self):
        from agent_system.environments.env_package.hiddenrule.hiddenrule import Item
        rule = Rule(family="count", device="button_D", n=1)
        self.world = make_fixture_world(rule)
        # 把 1--2 的门改成钥匙门,钥匙放房间 0
        self.world.doors[1] = Door(1, 2, "key", key_name="brass_key")
        self.world.items.append(Item(name="brass_key", room=0))
        self.state = initial_state(self.world)

    def test_locked_until_key_used(self):
        state = self.state
        state, _, _ = transition(self.world, state, "press button_D")
        state, _, _ = transition(self.world, state, "go to room 2")
        # 没钥匙: 去房间 3 不在 admissible 里
        assert "go to room 3" not in admissible_actions(self.world, state)
        state, _, valid = transition(self.world, state, "go to room 3")
        assert not valid
        # 回去拿钥匙 → 开门 → 通行
        state, _, _ = transition(self.world, state, "go to room 1")
        state, _, valid = transition(self.world, state, "pick up brass_key")
        assert valid
        state, _, _ = transition(self.world, state, "go to room 2")
        state, _, valid = transition(self.world, state, "use brass_key on door to room 3")
        assert valid
        assert "go to room 3" in admissible_actions(self.world, state)


# ---------------------------------------------------------------------------
# 验收 4: 环境接口行为
# ---------------------------------------------------------------------------

def test_invalid_action_is_noop_and_flagged():
    env = HiddenRuleEnv(HRGConfig())
    env.reset(seed=1)
    before = env.state
    obs, reward, done, info = env.step("cast fireball")
    assert not info["is_action_valid"]
    assert env.state == before
    assert "Nothing happens." in obs
    assert reward == 0.0


def test_admissible_actions_all_valid():
    env = HiddenRuleEnv(HRGConfig())
    env.reset(seed=3)
    for action in env.admissible_actions():
        state_before = env.state
        _, _, _, info = env.step(action)
        assert info["is_action_valid"], f"admissible action rejected: {action}"
        env.state = state_before  # 回滚继续测下一个
        env.steps -= 1


def test_horizon_termination():
    cfg = HRGConfig(max_steps=5)
    env = HiddenRuleEnv(cfg)
    env.reset(seed=2)
    done = False
    for _ in range(5):
        assert not done
        _, reward, done, info = env.step("look")
    assert done and not info["won"] and reward == 0.0


def test_latent_state_not_in_observation():
    """特权信息 (规则文本) 不得泄漏进观测——除非 agent 读到真便签"""
    env = HiddenRuleEnv(HRGConfig())
    for seed in range(10):
        obs, info = env.reset(seed=seed)
        assert info["rule_text"] not in obs


def test_true_note_reveals_rule():
    env = HiddenRuleEnv(HRGConfig(n_distractor_notes=0))
    env.reset(seed=5)
    true_note = next(n for n in env.world.notes if n.is_true)
    # 全知导航到便签所在房间 (BFS 简化: 直接瞬移状态)
    from dataclasses import replace
    env.state = replace(env.state, room=true_note.room)
    obs, _, _, info = env.step(f"read {true_note.name}")
    assert info["rule_text"] in obs
