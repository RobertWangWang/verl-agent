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
HiddenRule-Gym HRG-b 验收测试: 覆盖度 C 计算器 + 观测噪声/遮蔽旋钮

运行: pytest tests/test_hiddenrule_coverage.py -v
"""

import pytest

from agent_system.environments.env_package.hiddenrule.hiddenrule import (
    HiddenRuleEnv,
    HRGConfig,
    generate_world,
)
from agent_system.environments.env_package.hiddenrule.hiddenrule.coverage import (
    all_fields,
    calibrate_masks,
    coverage,
    enumerate_reachable,
)


def conj_world(seed=0, **cfg_kwargs):
    cfg = HRGConfig(rule_families=('conj',), key_door_prob=0.0, **cfg_kwargs)
    return generate_world(cfg, seed)


# ---------------------------------------------------------------------------
# 覆盖度 C (HRG-b 验收核心)
# ---------------------------------------------------------------------------

class TestCoverage:
    def test_full_fields_full_coverage_for_conj(self):
        """conj 族: latent = (room, devices),全字段无噪声 → C = 1.0 精确"""
        for seed in range(5):
            world = conj_world(seed)
            states = enumerate_reachable(world)
            c = coverage(world, frozenset(all_fields(world)), states=states)
            assert c == pytest.approx(1.0, abs=1e-9), f"seed {seed}: C={c}"

    def test_room_only_partial_coverage(self):
        world = conj_world(0)
        states = enumerate_reachable(world)
        c = coverage(world, frozenset({'room'}), states=states)
        assert 0.0 < c < 0.6

    def test_empty_mask_zero_coverage(self):
        world = conj_world(0)
        states = enumerate_reachable(world)
        assert coverage(world, frozenset(), states=states) == pytest.approx(0.0)

    def test_monotone_in_fields(self):
        """字段越多 C 越大 (信息单调性)"""
        world = conj_world(1)
        states = enumerate_reachable(world)
        fields = all_fields(world)
        prev = 0.0
        for k in range(len(fields) + 1):
            c = coverage(world, frozenset(fields[:k]), states=states)
            assert c >= prev - 1e-9
            prev = c

    def test_seq_family_latch_is_inherently_hidden(self):
        """seq 族: latch/进度不在任何观测字段里 → 全字段 C < 1 (这正是环境的部分可观测性)"""
        cfg = HRGConfig(rule_families=('seq',), key_door_prob=0.0)
        world = generate_world(cfg, 3)
        states = enumerate_reachable(world)
        c = coverage(world, frozenset(all_fields(world)), states=states)
        assert c < 1.0


class TestNoiseDecay:
    """HRG-b 验收: 加噪后 C 按预期衰减"""

    def test_flip_noise_reduces_coverage(self):
        world = conj_world(2)
        states = enumerate_reachable(world)
        mask = frozenset(all_fields(world))
        c0 = coverage(world, mask, states=states, flip_prob=0.0)
        c_low = coverage(world, mask, states=states, flip_prob=0.2, mc_samples=48, seed=1)
        c_high = coverage(world, mask, states=states, flip_prob=0.45, mc_samples=48, seed=1)
        assert c0 > c_low > c_high

    def test_mc_estimate_close_to_exact_at_zero_noise(self):
        world = conj_world(2)
        states = enumerate_reachable(world)
        mask = frozenset(all_fields(world))
        exact = coverage(world, mask, states=states, flip_prob=0.0)
        mc = coverage(world, mask, states=states, flip_prob=1e-9, mc_samples=8, seed=0)
        assert mc == pytest.approx(exact, abs=0.05)


class TestCalibrateMasks:
    def test_ladder_monotone_and_reaches_targets(self):
        for seed in (0, 1, 2):
            world = conj_world(seed)
            states = enumerate_reachable(world)
            ladder = calibrate_masks(world, states=states)
            prev_c = 0.0
            for target in (0.2, 0.4, 0.6, 0.8, 1.0):
                c = coverage(world, ladder[target], states=states)
                assert c >= target - 1e-9, f"seed {seed} target {target}: C={c}"
                assert c >= prev_c - 1e-9
                prev_c = c

    def test_ladder_masks_nested(self):
        world = conj_world(1)
        ladder = calibrate_masks(world)
        assert ladder[0.2] <= ladder[0.6] <= ladder[1.0]


# ---------------------------------------------------------------------------
# 渲染旋钮
# ---------------------------------------------------------------------------

class TestRenderKnobs:
    def test_default_config_renders_clean(self):
        env = HiddenRuleEnv(HRGConfig())
        obs, _ = env.reset(seed=0)
        assert 'Sensor panel' not in obs
        assert 'dimly lit' not in obs

    def test_sensor_channels_appear(self):
        env = HiddenRuleEnv(HRGConfig(n_sensor_channels=3))
        obs, _ = env.reset(seed=0)
        assert 'Sensor panel: P1=' in obs and 'P3=' in obs

    def test_p_obs_zero_hides_room_fields(self):
        env = HiddenRuleEnv(HRGConfig(p_obs=0.0))
        obs, _ = env.reset(seed=0)
        assert 'Devices here' not in obs
        assert 'dimly lit' in obs

    def test_noisy_obs_deterministic_replay(self):
        cfg = HRGConfig(p_obs=0.7, obs_flip_prob=0.3, n_sensor_channels=2)
        env1, env2 = HiddenRuleEnv(cfg), HiddenRuleEnv(cfg)
        obs1, _ = env1.reset(seed=9)
        obs2, _ = env2.reset(seed=9)
        assert obs1 == obs2
        for action in ("look", "look", "look"):
            o1, _, _, _ = env1.step(action)
            o2, _, _, _ = env2.step(action)
            assert o1 == o2

    def test_flip_changes_displayed_state_not_latent(self):
        """flip=1.0: 文本必显示错误读数,但 latent state 不受影响"""
        cfg = HRGConfig(obs_flip_prob=1.0, rule_families=('conj',), key_door_prob=0.0)
        env = HiddenRuleEnv(cfg)
        env.reset(seed=4)
        # 找一个 agent 所在房间的 lever
        world, state = env.world, env.state
        lever = next((d for d in world.devices
                      if d.room == state.room and d.kind == 'lever'), None)
        if lever is None:
            pytest.skip("seed 4 起始房间无 lever")
        obs, _, _, info = env.step(f"toggle {lever.name}")
        true_state = info['latent_state']['device_states'][lever.name]
        assert true_state == 1  # latent 真的 toggle 了
        # 文本层 flip=1.0 → 二值 lever 必显示相反读数 (down)
        assert f"{lever.name} is down" in obs

    def test_noise_does_not_affect_dynamics(self):
        """噪声只在文本层: 同 seed 下,有噪/无噪的 latent 轨迹完全一致"""
        clean = HiddenRuleEnv(HRGConfig())
        noisy = HiddenRuleEnv(HRGConfig(p_obs=0.5, obs_flip_prob=0.4, n_sensor_channels=2))
        _, info_c = clean.reset(seed=7)
        _, info_n = noisy.reset(seed=7)
        assert info_c['latent_state'] == info_n['latent_state']
        for action in ("look", "look"):
            _, _, _, ic = clean.step(action)
            _, _, _, inn = noisy.step(action)
            assert ic['latent_state'] == inn['latent_state']
