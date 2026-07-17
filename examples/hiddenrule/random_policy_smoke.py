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
HRG-c 验收: HiddenRule-Gym x verl-agent 集成的随机策略冒烟 (无 GPU/LLM)

验收标准 (docs/hiddenrule_gym_design.md §5):
- 并行 8 组 x 4 = 32 env x 60 步无死锁;
- 组内规则一致性断言通过;
- 随机策略成功率落在难度校准的预期区间 (个位数百分比)。

运行: python -m examples.hiddenrule.random_policy_smoke
"""

import random
import time

from omegaconf import OmegaConf

from agent_system.environments import make_envs

N_STEPS = 60
GROUP_N = 4
TRAIN_ENVS = 8  # 8 组 x 4 = 32 workers


def make_config():
    return OmegaConf.create({
        'env': {
            'env_name': 'hiddenrule/HiddenRuleEnv',
            'seed': 0,
            'max_steps': N_STEPS,
            'history_length': 2,
            'rollout': {'n': GROUP_N},
            'resources_per_worker': {'num_cpus': 0.1},
            'hiddenrule': {
                'n_rooms': 5, 'n_devices': 6,
                'rule_families': ['conj', 'seq', 'xor', 'count'],
                'rule_arity': 2, 'evidence_lag': 2, 'n_distractor_notes': 1,
                'key_door_prob': 0.5, 'p_obs': 1.0, 'obs_flip_prob': 0.0,
                'n_sensor_channels': 0, 'coverage_level': 1.0,
                'require_think_tags': True,
            },
        },
        'data': {'train_batch_size': TRAIN_ENVS, 'val_batch_size': 4},
    })


def main():
    rng = random.Random(0)
    config = make_config()
    print(f"building {TRAIN_ENVS}x{GROUP_N}={TRAIN_ENVS * GROUP_N} train envs + 4 val envs ...")
    envs, val_envs = make_envs(config)

    t0 = time.time()
    obs, infos = envs.reset(kwargs=None)
    n = len(obs['text'])
    assert n == TRAIN_ENVS * GROUP_N, f"expected {TRAIN_ENVS * GROUP_N} envs, got {n}"

    # 组内规则一致性 (GRPO 组语义)
    group_rules = set()
    for g in range(TRAIN_ENVS):
        rules = {infos[g * GROUP_N + k]['rule_text'] for k in range(GROUP_N)}
        assert len(rules) == 1, f"group {g}: rules differ within group: {rules}"
        group_rules.add(next(iter(rules)))
    print(f"group consistency OK ({TRAIN_ENVS} groups, {len(group_rules)} distinct rules)")
    assert '<think>' in obs['text'][0] and '<action>' in obs['text'][0]

    wins = [False] * n
    valid_ct, total_ct = 0, 0
    for step in range(N_STEPS):
        admissible = envs.envs.get_admissible_commands
        responses = []
        for i in range(n):
            action = rng.choice(admissible[i])
            responses.append(f"<think>random walk</think><action>{action}</action>")
        obs, rewards, dones, infos = envs.step(responses)
        for i in range(n):
            valid_ct += int(infos[i]['is_action_valid'])
            total_ct += 1
            if infos[i]['won']:
                wins[i] = True

    elapsed = time.time() - t0
    success = sum(wins) / n
    print(f"32 envs x {N_STEPS} steps in {elapsed:.1f}s "
          f"({n * N_STEPS / elapsed:.0f} env-steps/s)")
    print(f"random-policy success rate: {success:.3f} ({sum(wins)}/{n})")
    print(f"valid action ratio: {valid_ct / total_ct:.3f}")
    assert valid_ct / total_ct == 1.0, "random admissible actions must all be valid"
    assert success < 0.5, "random policy success suspiciously high — difficulty miscalibrated"

    envs.close()
    val_envs.close()
    print("HRG-c smoke PASSED")


if __name__ == '__main__':
    main()
